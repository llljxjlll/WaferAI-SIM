#include "memory/hbm_mem_wire.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {

constexpr unsigned kMagic = 0xD7;
constexpr unsigned kVersion = 1;

void Put(MemWireFlit &w, int lo, int bits, uint64_t value) {
    sc_dt::sc_bv<64> encoded(value);
    w.range(lo + bits - 1, lo) = encoded.range(bits - 1, 0);
}

uint64_t Get(const MemWireFlit &w, int lo, int bits) {
    return w.range(lo + bits - 1, lo).to_uint64();
}

MemWireFlit Base(MemFlitKind kind, int txid) {
    if (txid < 0)
        throw std::runtime_error("SerializeMemMsg: txid must be non-negative");
    MemWireFlit w;
    w = 0;
    Put(w, 0, 8, kMagic);
    Put(w, 8, 4, kVersion);
    Put(w, 12, 4, (unsigned)kind);
    Put(w, 16, 32, (uint32_t)txid);
    return w;
}

MemFlitKind Kind(const MemWireFlit &w) {
    if (Get(w, 0, 8) != kMagic || Get(w, 8, 4) != kVersion)
        throw std::runtime_error("DeserializeMemMsg: bad magic/version");
    unsigned k = (unsigned)Get(w, 12, 4);
    if (k > (unsigned)MemFlitKind::kResponse)
        throw std::runtime_error("DeserializeMemMsg: unknown flit kind");
    return (MemFlitKind)k;
}

void ValidateCommon(const MemMsg &m) {
    if (m.txid < 0 || m.source_core < 0 || m.source_core > 0xffff ||
        m.home_die < 0 || m.home_die > 0xfff || m.stack_id < 0 ||
        m.stack_id > 0xfff || m.channel_id < 0 || m.channel_id > 0xfff)
        throw std::runtime_error("SerializeMemMsg: identifier out of wire range");
    if (m.length_bytes < 0 || m.length_bytes > kMemMsgMaxPayloadBytes)
        throw std::runtime_error("SerializeMemMsg: length out of range");
    if (!m.byte_enable.empty() &&
        (int)m.byte_enable.size() != m.length_bytes)
        throw std::runtime_error(
            "SerializeMemMsg: byte_enable size must equal length");
}

void PutData(MemWireFlit &w, const MemMsg &m, int seq, int offset,
             int valid) {
    Put(w, 48, 16, seq);
    Put(w, 64, 8, valid);
    Put(w, 72, 1, offset + valid == m.length_bytes);
    uint16_t enables = 0;
    for (int i = 0; i < valid; ++i) {
        bool enabled = m.byte_enable.empty() || m.byte_enable[offset + i] != 0;
        if (enabled)
            enables |= (uint16_t)(1u << i);
        Put(w, 89 + i * 8, 8, m.payload[offset + i]);
    }
    Put(w, 73, 16, enables);
    Put(w, 201, 16, m.source_core);
    Put(w, 217, 12, m.home_die);
    Put(w, 229, 12, m.stack_id);
    Put(w, 241, 12, m.channel_id);
}

} // namespace

bool IsMemWireFlit(const MemWireFlit &wire) {
    return Get(wire, 0, 8) == kMagic && Get(wire, 8, 4) == kVersion &&
           Get(wire, 12, 4) <= (unsigned)MemFlitKind::kResponse;
}

MemFlitRoute InspectMemWireFlit(const MemWireFlit &wire) {
    MemFlitRoute r;
    r.kind = Kind(wire);
    r.txid = (int)(uint32_t)Get(wire, 16, 32);
    r.request_direction = r.kind == MemFlitKind::kRequest ||
                          r.kind == MemFlitKind::kWriteData;
    if (r.kind == MemFlitKind::kRequest) {
        r.source_core = (int)Get(wire, 48, 16);
        r.home_die = (int)Get(wire, 64, 12);
        r.stack_id = (int)Get(wire, 76, 12);
        r.channel_id = (int)Get(wire, 88, 12);
        r.transaction_end = Get(wire, 180, 1) == 0;
    } else if (r.kind == MemFlitKind::kResponse) {
        r.source_core = (int)Get(wire, 48, 16);
        r.transaction_end = true;
    } else {
        r.sequence = (int)Get(wire, 48, 16);
        r.source_core = (int)Get(wire, 201, 16);
        r.home_die = (int)Get(wire, 217, 12);
        r.stack_id = (int)Get(wire, 229, 12);
        r.channel_id = (int)Get(wire, 241, 12);
        r.transaction_end = Get(wire, 72, 1) != 0;
    }
    return r;
}

std::vector<MemWireFlit> CanonicalizeMemWireFlits(
    const std::vector<MemWireFlit> &wire) {
    std::vector<MemWireFlit> ordered = wire;
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const MemWireFlit &a, const MemWireFlit &b) {
        const auto ra = InspectMemWireFlit(a);
        const auto rb = InspectMemWireFlit(b);
        auto rank = [](MemFlitKind k) {
            if (k == MemFlitKind::kRequest) return 0;
            if (k == MemFlitKind::kWriteData ||
                k == MemFlitKind::kReadData) return 1;
            return 2;
        };
        const int ar = rank(ra.kind), br = rank(rb.kind);
        if (ar != br) return ar < br;
        if (ar == 1) return ra.sequence < rb.sequence;
        return false;
    });
    return ordered;
}

std::vector<MemWireFlit> SerializeMemMsg(const MemMsg &m) {
    ValidateCommon(m);
    std::vector<MemWireFlit> out;
    if (m.message_type == MemMessageType::kRequest) {
        if (m.command == MemCommand::kWrite &&
            (int)m.payload.size() != m.length_bytes)
            throw std::runtime_error(
                "SerializeMemMsg: write payload size must equal length");
        if (m.command == MemCommand::kRead && !m.payload.empty())
            throw std::runtime_error(
                "SerializeMemMsg: read request must not carry payload");
        MemWireFlit req = Base(MemFlitKind::kRequest, m.txid);
        Put(req, 48, 16, m.source_core);
        Put(req, 64, 12, m.home_die);
        Put(req, 76, 12, m.stack_id);
        Put(req, 88, 12, m.channel_id);
        Put(req, 100, 64, m.address);
        Put(req, 164, 16, m.length_bytes);
        Put(req, 180, 1, m.command == MemCommand::kWrite);
        Put(req, 181, 1, 1);
        out.push_back(req);
        if (m.command == MemCommand::kWrite) {
            for (int off = 0, seq = 0; off < m.length_bytes;
                 off += kMemFlitPayloadBytes, ++seq) {
                int valid = std::min(kMemFlitPayloadBytes,
                                     m.length_bytes - off);
                MemWireFlit data = Base(MemFlitKind::kWriteData, m.txid);
                PutData(data, m, seq, off, valid);
                out.push_back(data);
            }
        }
    } else {
        if (m.command == MemCommand::kRead && m.status == 0 &&
            (int)m.payload.size() != m.length_bytes)
            throw std::runtime_error(
                "SerializeMemMsg: read response payload size mismatch");
        if (m.command == MemCommand::kRead && m.status == 0) {
            for (int off = 0, seq = 0; off < m.length_bytes;
                 off += kMemFlitPayloadBytes, ++seq) {
                int valid = std::min(kMemFlitPayloadBytes,
                                     m.length_bytes - off);
                MemWireFlit data = Base(MemFlitKind::kReadData, m.txid);
                PutData(data, m, seq, off, valid);
                out.push_back(data);
            }
        }
        MemWireFlit resp = Base(MemFlitKind::kResponse, m.txid);
        Put(resp, 48, 16, m.source_core);
        Put(resp, 64, 8, m.status);
        Put(resp, 72, 16, m.length_bytes);
        Put(resp, 88, 1, 1);
        Put(resp, 89, 1, m.command == MemCommand::kWrite);
        out.push_back(resp);
    }
    return out;
}

MemMsg DeserializeMemMsg(const std::vector<MemWireFlit> &wire) {
    if (wire.empty())
        throw std::runtime_error("DeserializeMemMsg: empty transaction");
    MemMsg m;
    MemFlitKind first = Kind(wire.front());
    m.txid = (int)(uint32_t)Get(wire.front(), 16, 32);
    size_t data_begin = 0, data_end = 0;
    MemFlitKind data_kind = MemFlitKind::kWriteData;

    if (first == MemFlitKind::kRequest) {
        m.message_type = MemMessageType::kRequest;
        m.source_core = (int)Get(wire.front(), 48, 16);
        m.home_die = (int)Get(wire.front(), 64, 12);
        m.stack_id = (int)Get(wire.front(), 76, 12);
        m.channel_id = (int)Get(wire.front(), 88, 12);
        m.address = Get(wire.front(), 100, 64);
        m.length_bytes = (int)Get(wire.front(), 164, 16);
        m.command = Get(wire.front(), 180, 1) ? MemCommand::kWrite
                                              : MemCommand::kRead;
        if (Get(wire.front(), 181, 1) != 1)
            throw std::runtime_error("DeserializeMemMsg: request header not final");
        data_begin = 1;
        data_end = wire.size();
        data_kind = MemFlitKind::kWriteData;
        if (m.command == MemCommand::kRead && wire.size() != 1)
            throw std::runtime_error(
                "DeserializeMemMsg: read request has unexpected data flits");
    } else {
        if (Kind(wire.back()) != MemFlitKind::kResponse)
            throw std::runtime_error(
                "DeserializeMemMsg: response transaction lacks completion");
        const MemWireFlit &resp = wire.back();
        if ((int)(uint32_t)Get(resp, 16, 32) != m.txid)
            throw std::runtime_error("DeserializeMemMsg: txid mismatch");
        m.message_type = MemMessageType::kResponse;
        m.source_core = (int)Get(resp, 48, 16);
        m.status = (int)Get(resp, 64, 8);
        m.length_bytes = (int)Get(resp, 72, 16);
        m.command = Get(resp, 89, 1) ? MemCommand::kWrite : MemCommand::kRead;
        data_begin = 0;
        data_end = wire.size() - 1;
        data_kind = MemFlitKind::kReadData;
    }

    for (size_t i = data_begin, seq = 0; i < data_end; ++i, ++seq) {
        const MemWireFlit &d = wire[i];
        if (Kind(d) != data_kind ||
            (int)(uint32_t)Get(d, 16, 32) != m.txid ||
            Get(d, 48, 16) != seq)
            {
                std::ostringstream os;
                os << "DeserializeMemMsg: data kind/txid/sequence mismatch"
                   << " index=" << i << " expected_kind=" << (int)data_kind
                   << " got_kind=" << (int)Kind(d)
                   << " expected_txid=" << m.txid
                   << " got_txid=" << (int)(uint32_t)Get(d, 16, 32)
                   << " expected_seq=" << seq
                   << " got_seq=" << Get(d, 48, 16);
                throw std::runtime_error(os.str());
            }
        int valid = (int)Get(d, 64, 8);
        if (valid <= 0 || valid > kMemFlitPayloadBytes)
            throw std::runtime_error(
                "DeserializeMemMsg: invalid data flit length");
        uint16_t enables = (uint16_t)Get(d, 73, 16);
        for (int b = 0; b < valid; ++b) {
            m.payload.push_back((uint8_t)Get(d, 89 + b * 8, 8));
            m.byte_enable.push_back((enables & (1u << b)) ? 0xff : 0);
        }
        bool end = Get(d, 72, 1) != 0;
        if (end != (i + 1 == data_end))
            throw std::runtime_error(
                "DeserializeMemMsg: invalid data end marker");
    }
    if (m.command == MemCommand::kWrite &&
        m.message_type == MemMessageType::kRequest &&
        (int)m.payload.size() != m.length_bytes)
        throw std::runtime_error(
            "DeserializeMemMsg: incomplete write payload");
    if (m.command == MemCommand::kRead &&
        m.message_type == MemMessageType::kResponse && m.status == 0 &&
        (int)m.payload.size() != m.length_bytes)
        throw std::runtime_error(
            "DeserializeMemMsg: incomplete read payload");
    m.is_end = true;
    return m;
}
