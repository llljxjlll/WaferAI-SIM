#include "prims/moe_signed_router_npu_prim.h"

#include "memory/sram/sram_access_unit.h"
#include "memory/sram/sram_storage.h"
#include "utils/prim_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

REGISTER_PRIM(moe_score_weighted_forward,
              PrimId::MOE_SCORE_WEIGHTED_FORWARD);
REGISTER_PRIM(moe_score_weight_backward,
              PrimId::MOE_SCORE_WEIGHT_BACKWARD);

namespace {
struct Span { uint64_t address; uint64_t bytes; };
struct Layout {
    uint64_t k, h, e;
    Span route, score, returned, dcombined, combined, dscore, dexpert;
};

uint64_t Parameter(const NpuBase &prim, const char *name) {
    const auto it = prim.param_value.find(name);
    if (it == prim.param_value.end() || it->second < 0)
        throw std::invalid_argument(prim.name + " missing/nonnegative " + name);
    return static_cast<uint64_t>(it->second);
}

uint64_t Multiply(uint64_t x, uint64_t y, const char *field) {
    if (y != 0 && x > UINT64_MAX / y)
        throw std::overflow_error(std::string(field) + " overflow");
    return x * y;
}

void CheckSpan(Span s, const char *field) {
    if (s.address % 16 != 0 || s.bytes == 0 ||
        s.address > UINT16_MAX || s.bytes > uint64_t{UINT16_MAX} + 1 - s.address)
        throw std::invalid_argument(std::string(field) +
                                    " requires independent aligned 16-bit SRAM span");
}

Layout CheckLayout(const NpuBase &p, bool backward) {
    const size_t required = backward ? 7 : 5;
    if (p.datatype != FP16 || p.param_value.size() != required ||
        p.inp_offset < 0 || p.data_offset < 0 || p.out_offset < 0)
        throw std::invalid_argument(p.name + " strict multiword FP16/profile/offset required");
    const uint64_t k = Parameter(p, "K");
    const uint64_t h = Parameter(p, "H");
    const uint64_t e = Parameter(p, "E");
    const uint64_t route_bytes = Parameter(p, "ROUTE_BYTES");
    if (!k || !h || !e ||
        k > ((uint64_t{1} << 30) - 1) ||
        h > ((uint64_t{1} << 30) - 1) ||
        e > ((uint64_t{1} << 30) - 1) ||
        route_bytes != Multiply(20, k, "route 5xINT32"))
        throw std::invalid_argument(p.name + " must consume all five INT32 route fields");
    const uint64_t score_bytes = Multiply(2, Multiply(k,e,"KxE"),"score bytes");
    const uint64_t hidden_bytes = Multiply(2, Multiply(k,h,"KxH"),"expert bytes");
    Layout v{k,h,e,
        {Parameter(p,"ROUTE_ADDRESS"), route_bytes},
        {static_cast<uint64_t>(p.inp_offset), score_bytes},
        {static_cast<uint64_t>(p.data_offset), hidden_bytes},
        {static_cast<uint64_t>(p.out_offset), hidden_bytes},
        {static_cast<uint64_t>(p.out_offset), hidden_bytes},
        {},{}};
    std::vector<Span> spans{v.route,v.score,v.returned,v.combined};
    if (backward) {
        v.dscore = {Parameter(p,"DSCORE_ADDRESS"),score_bytes};
        v.dexpert = {Parameter(p,"DEXPERT_ADDRESS"),hidden_bytes};
        spans.push_back(v.dscore);
        spans.push_back(v.dexpert);
    }
    for (const auto &span : spans) CheckSpan(span, p.name.c_str());
    for (size_t i=0; i<spans.size(); ++i)
        for (size_t j=i+1; j<spans.size(); ++j)
            if (spans[i].address < spans[j].address+spans[j].bytes &&
                spans[j].address < spans[i].address+spans[i].bytes)
                throw std::invalid_argument(p.name + " score/route/expert/gradient buffers overlap");
    return v;
}

int HalfElements(uint64_t bytes) {
    if (bytes % 2 != 0 || bytes / 2 > INT_MAX)
        throw std::overflow_error("router input bytes exceed NpuBase element ABI");
    return static_cast<int>(bytes/2);
}

uint32_t Le32(const std::vector<uint8_t> &bytes, size_t offset) {
    return static_cast<uint32_t>(bytes.at(offset)) |
           (static_cast<uint32_t>(bytes.at(offset+1)) << 8) |
           (static_cast<uint32_t>(bytes.at(offset+2)) << 16) |
           (static_cast<uint32_t>(bytes.at(offset+3)) << 24);
}

std::vector<uint8_t> Read(TaskCoreContext &ctx, Span s) {
    sram::Request req;
    req.initiator=sram::Initiator::kCompute;
    req.command=sram::Command::kRead;
    req.address=s.address;
    req.size_bytes=s.bytes;
    const auto bytes=ctx.sram_access->Access(req).payload;
    if (bytes.size()!=s.bytes)
        throw std::runtime_error("router SRAM read returned truncated payload");
    return bytes;
}

void Write(TaskCoreContext &ctx, Span s, std::vector<uint8_t> bytes) {
    if (bytes.size()!=s.bytes)
        throw std::runtime_error("router FP16 output must write complete owned SRAM span");
    sram::Request req;
    req.initiator=sram::Initiator::kCompute;
    req.command=sram::Command::kWrite;
    req.address=s.address;
    req.size_bytes=s.bytes;
    req.payload=std::move(bytes);
    ctx.sram_access->Access(req);
}

float HalfToFloat(uint16_t half) {
    const uint32_t sign=static_cast<uint32_t>(half & 0x8000u)<<16;
    const uint32_t fraction=half & 0x03ffu;
    const uint32_t exponent=(half>>10) & 0x1fu;
    uint32_t bits;
    if (exponent==0) {
        if (fraction==0) bits=sign;
        else {
            uint32_t normal=fraction;
            uint32_t e=113u;
            while ((normal & 0x0400u)==0) {normal <<= 1; --e;}
            bits=sign | (e<<23) | ((normal & 0x03ffu)<<13);
        }
    } else if (exponent==31) bits=sign | 0x7f800000u | (fraction<<13);
    else bits=sign | ((exponent+112u)<<23) | (fraction<<13);
    float value;
    std::memcpy(&value,&bits,sizeof(value));
    return value;
}

uint16_t FloatToHalf(float value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("router FP16 score/gradient output must be finite");
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    const uint16_t sign=static_cast<uint16_t>((bits>>16)&0x8000u);
    const int32_t exp=static_cast<int32_t>((bits>>23)&0xffu)-127;
    const uint32_t mant=bits & 0x7fffffu;
    if (exp>15) throw std::overflow_error("router FP16 output overflows");
    if (exp>=-14) {
        const uint32_t rounded=mant + 0x00000fffu + ((mant>>13)&1u);
        uint32_t h=static_cast<uint32_t>(exp+15)<<10;
        h += rounded>>13;
        if (h>=0x7c00u) throw std::overflow_error("router FP16 rounded output overflows");
        return static_cast<uint16_t>(sign|h);
    }
    if (exp<-24) return sign;
    const uint32_t normal=mant|0x800000u;
    const uint32_t shift=static_cast<uint32_t>(-14-exp)+13u;
    const uint32_t half=normal>>shift;
    const uint32_t discarded=normal & ((uint32_t{1}<<shift)-1);
    const uint32_t midpoint=uint32_t{1}<<(shift-1);
    return static_cast<uint16_t>(sign| (half + (discarded>midpoint ||
                                                (discarded==midpoint && (half&1)))));
}

std::vector<float> UnpackHalf(const std::vector<uint8_t> &bytes) {
    if (bytes.size()%2) throw std::invalid_argument("router FP16 SRAM span odd");
    std::vector<float> result;
    result.reserve(bytes.size()/2);
    for (size_t i=0;i<bytes.size();i+=2) {
        const auto value=HalfToFloat(static_cast<uint16_t>(bytes[i] |
                                     (static_cast<uint16_t>(bytes[i+1])<<8)));
        if (!std::isfinite(value))
            throw std::invalid_argument("router FP16 SRAM operand contains NaN/Inf");
        result.push_back(value);
    }
    return result;
}

std::vector<uint8_t> PackHalf(const std::vector<float> &values) {
    std::vector<uint8_t> result;
    result.reserve(values.size()*2);
    for (float value:values) {
        const uint16_t bits=FloatToHalf(value);
        result.push_back(static_cast<uint8_t>(bits));
        result.push_back(static_cast<uint8_t>(bits>>8));
    }
    return result;
}

std::vector<uint64_t> ReadRoute(TaskCoreContext &ctx, const Layout &layout) {
    const auto blob=Read(ctx,layout.route);
    std::vector<std::array<uint32_t,5>> route;
    route.reserve(layout.k);
    std::vector<uint64_t> counts(layout.e);
    for (size_t row=0;row<layout.k;++row) {
        const size_t at=row*20;
        const std::array<uint32_t,5> fields{{Le32(blob,at),Le32(blob,at+4),
             Le32(blob,at+8),Le32(blob,at+12),Le32(blob,at+16)}};
        if (fields[0]!=row || fields[1]!=0 || fields[2]>=layout.e ||
            fields[3]!=fields[2])
            throw std::invalid_argument("router physical INT32 token/source/EP owner differs from signed top1 source");
        ++counts[fields[2]];
        route.push_back(fields);
    }
    uint64_t prefix=0;
    std::vector<uint64_t> starts(layout.e);
    for (size_t expert=0;expert<layout.e;++expert) {
        starts[expert]=prefix;
        prefix+=counts[expert];
    }
    if (prefix!=layout.k)
        throw std::invalid_argument("router route groups drop a source token");
    std::vector<std::vector<bool>> slots(layout.e);
    for (size_t expert=0;expert<layout.e;++expert)
        slots[expert].resize(counts[expert]);
    std::vector<uint64_t> row_to_group(layout.k);
    for (size_t row=0;row<layout.k;++row) {
        const auto expert=route[row][2];
        const auto slot=route[row][4];
        if (slot>=counts[expert] || slots[expert][slot])
            throw std::invalid_argument("router route duplicates or exceeds selected expert home slot");
        slots[expert][slot]=true;
        row_to_group[row]=starts[expert]+slot;
    }
    for (const auto &expert_slots:slots)
        if (std::any_of(expert_slots.begin(),expert_slots.end(),
                        [](bool used){return !used;}))
            throw std::invalid_argument("router route fails complete expert home slot coverage");
    // Map token row to grouped expert row and selected expert in a single
    // physical read. Token/expert route data is never supplied as a literal.
    for (size_t row=0;row<layout.k;++row)
        row_to_group[row] |= uint64_t{route[row][2]}<<32;
    return row_to_group;
}

void RequirePayload(TaskCoreContext &ctx) {
    if (ctx.sram_access==nullptr || ctx.sram_storage==nullptr ||
        !ctx.sram_storage->payload_mode())
        throw std::logic_error("router native requires real SRAM payload mode");
}

void ExecuteForward(TaskCoreContext &ctx,const Layout &v) {
    RequirePayload(ctx);
    const auto routes=ReadRoute(ctx,v);
    const auto scores=UnpackHalf(Read(ctx,v.score));
    const auto returned=UnpackHalf(Read(ctx,v.returned));
    std::vector<float> combined(v.k*v.h);
    for (size_t row=0;row<v.k;++row) {
        const size_t expert=routes[row]>>32;
        const size_t home_slot=routes[row]&0xffffffffu;
        const float score=scores[row*v.e+expert];
        for (size_t col=0;col<v.h;++col)
            combined[row*v.h+col]=score*returned[home_slot*v.h+col];
    }
    Write(ctx,v.combined,PackHalf(combined));
    std::cout << "[MOE_SIGNED_ROUTER] stage=forward rows=" << v.k
              << " hidden=" << v.h << " experts=" << v.e
              << " route_read_bytes=" << v.route.bytes
              << " fp16_read_bytes=" << v.score.bytes + v.returned.bytes
              << " combined_write_bytes=" << v.combined.bytes
              << " pass=1\n";
}

void ExecuteBackward(TaskCoreContext &ctx,const Layout &v) {
    RequirePayload(ctx);
    const auto routes=ReadRoute(ctx,v);
    const auto scores=UnpackHalf(Read(ctx,v.score));
    const auto returned=UnpackHalf(Read(ctx,v.returned));
    const auto upstream=UnpackHalf(Read(ctx,v.dcombined));
    std::vector<float> dscore(v.k*v.e);
    std::vector<float> dexpert(v.k*v.h);
    for (size_t row=0;row<v.k;++row) {
        const size_t expert=routes[row]>>32;
        const size_t home_slot=routes[row]&0xffffffffu;
        const float score=scores[row*v.e+expert];
        float dot=0;
        for (size_t col=0;col<v.h;++col) {
            const float dy=upstream[row*v.h+col];
            dot+=dy*returned[home_slot*v.h+col];
            dexpert[home_slot*v.h+col]=score*dy;
        }
        dscore[row*v.e+expert]=dot;
    }
    Write(ctx,v.dscore,PackHalf(dscore));
    Write(ctx,v.dexpert,PackHalf(dexpert));
    std::cout << "[MOE_SIGNED_ROUTER] stage=backward rows=" << v.k
              << " hidden=" << v.h << " experts=" << v.e
              << " route_read_bytes=" << v.route.bytes
              << " fp16_read_bytes="
              << v.score.bytes + v.returned.bytes + v.dcombined.bytes
              << " dscore_write_bytes=" << v.dscore.bytes
              << " dexpert_write_bytes=" << v.dexpert.bytes
              << " pass=1\n";
}

void StrictWire(const NpuBase &p) {
    if (prim_wire::LegacyCompatibilityEnabled())
        throw std::invalid_argument(p.name + " needs strict multiword Prim wire");
}
} // namespace

moe_score_weighted_forward::moe_score_weighted_forward() {
    name="moe_score_weighted_forward";
    datatype=FP16;
    skip_input=skip_output=true;
    param_name={"E","H","K","ROUTE_ADDRESS","ROUTE_BYTES"};
}

void moe_score_weighted_forward::initialize() {
    const Layout v=CheckLayout(*this,false);
    data_size_input={HalfElements(v.route.bytes), HalfElements(v.score.bytes),
                     HalfElements(v.returned.bytes)};
    data_chunk={{"output",HalfElements(v.combined.bytes)}};
}

void moe_score_weighted_forward::taskCore(TaskCoreContext &ctx,string,
    u_int64_t &dram,u_int64_t &exu,u_int64_t &sfu,u_int64_t &vec) {
    const Layout v=CheckLayout(*this,false);
    ExecuteForward(ctx,v);
    dram=sfu=vec=0;
    exu=Multiply(2,Multiply(v.k,v.h,"router FWD cells"),"router FWD EXU");
}

vector<sc_bv<128>> moe_score_weighted_forward::serialize() {
    StrictWire(*this);
    CheckLayout(*this,false);
    return NpuBase::serialize();
}

void moe_score_weighted_forward::deserialize(vector<sc_bv<128>> wire) {
    StrictWire(*this);
    NpuBase::deserialize(std::move(wire));
    CheckLayout(*this,false);
}

moe_score_weight_backward::moe_score_weight_backward() {
    name="moe_score_weight_backward";
    datatype=FP16;
    skip_input=skip_output=true;
    param_name={"DEXPERT_ADDRESS","DSCORE_ADDRESS","E","H","K",
                "ROUTE_ADDRESS","ROUTE_BYTES"};
}

void moe_score_weight_backward::initialize() {
    const Layout v=CheckLayout(*this,true);
    data_size_input={HalfElements(v.route.bytes), HalfElements(v.score.bytes),
                     HalfElements(v.returned.bytes),HalfElements(v.dcombined.bytes)};
    // NpuBase requires a formal output chunk even when manual SRAM compute
    // writes its two independent physical outputs and skip_output is true.
    data_chunk={{"output",HalfElements(v.dscore.bytes)}};
}

void moe_score_weight_backward::taskCore(TaskCoreContext &ctx,string,
    u_int64_t &dram,u_int64_t &exu,u_int64_t &sfu,u_int64_t &vec) {
    const Layout v=CheckLayout(*this,true);
    ExecuteBackward(ctx,v);
    dram=sfu=0;
    exu=Multiply(2,Multiply(v.k,v.h,"router BWD cells"),"router BWD EXU");
    vec=Multiply(v.k,v.h,"router BWD vector");
}

vector<sc_bv<128>> moe_score_weight_backward::serialize() {
    StrictWire(*this);
    CheckLayout(*this,true);
    return NpuBase::serialize();
}

void moe_score_weight_backward::deserialize(vector<sc_bv<128>> wire) {
    StrictWire(*this);
    NpuBase::deserialize(std::move(wire));
    CheckLayout(*this,true);
}
