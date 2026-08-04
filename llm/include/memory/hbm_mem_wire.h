#pragma once
// R1：独立 MEM wire 协议（MEM_REQ/WDATA 打包进请求，MEM_RESP/RDATA 打包进响应）。
// 详见 notes/extensions/DRAM/HBM建模计划.md（修订版）四.3。
//
// 现有通用 Msg.offset_ 在 wire 上只有 8 bit（M_D_OFFSET，macros.h:102），装不下 64-bit
// 物理地址，且 REQUEST/ACK/DATA 已经承担 send/collective 协议语义，不能借用。这里是
// 完全独立的编解码，序列化到固定宽度 sc_bv，静态位宽检查方式与 msg_utils.cpp 一致。
#include <cstdint>
#include <systemc>
#include <vector>

enum class MemCommand : uint8_t { kRead = 0, kWrite = 1 };
enum class MemMessageType : uint8_t { kRequest = 0, kResponse = 1 };
enum class MemFlitKind : uint8_t {
    kRequest = 0,
    kWriteData = 1,
    kReadData = 2,
    kResponse = 3,
};

// 数据 flit 的尾部保留逐 flit 路由元数据。14 B payload + 16-bit byte-enable
// 正好给 source/home/stack/channel 留出空间，因此 WDATA/RDATA 即使与 header 分离、
// 在 router 中被逐拍仲裁，也能独立决定下一跳。
constexpr int kMemFlitPayloadBytes = 14;
constexpr int kMemMsgMaxPayloadBytes = 65535;
constexpr int kMemWireTotalBits = 256;
using MemWireFlit = sc_dt::sc_bv<kMemWireTotalBits>;

struct MemMsg {
    int txid = -1;
    MemMessageType message_type = MemMessageType::kRequest;
    int source_core = -1;
    int home_die = -1;
    int stack_id = -1;
    int channel_id = -1;
    uint64_t address = 0; // DecodeAddress 给出的 local_address（channel 内地址）
    MemCommand command = MemCommand::kRead;
    int length_bytes = 0; // <= kMemMsgMaxPayloadBytes
    bool is_end = true;
    int status = 0; // 0=OK；响应用；请求恒为 0
    std::vector<uint8_t> payload; // 写请求的写数据 / 读响应的读数据
    // 每个 payload byte 一项：0=禁用，非零=启用。为空表示全部启用。
    std::vector<uint8_t> byte_enable;
};

struct MemFlitRoute {
    MemFlitKind kind = MemFlitKind::kRequest;
    int txid = -1;
    int source_core = -1;
    int home_die = -1;
    int stack_id = -1;
    int channel_id = -1;
    int sequence = -1;
    bool request_direction = true;
    bool transaction_end = false;
};

// Router/D2D 的轻量识别接口；不需要先组装整笔 transaction。
bool IsMemWireFlit(const MemWireFlit &wire);
MemFlitRoute InspectMemWireFlit(const MemWireFlit &wire);
std::vector<MemWireFlit> CanonicalizeMemWireFlits(
    const std::vector<MemWireFlit> &wire);

// 逻辑 transaction 按消息类型展开为 REQ + WDATA* 或 RDATA* + RESP。每个物理 flit
// 恰好 256 bit，可直接进入现有 RouterUnit 的 sc_bv<256> 信道。
std::vector<MemWireFlit> SerializeMemMsg(const MemMsg &msg);
MemMsg DeserializeMemMsg(const std::vector<MemWireFlit> &wire);
