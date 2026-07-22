#pragma once
// V0b-2C1：workload id 归一化 + 结构校验（与物理 HOST/MemInterface 无关，可纯函数测试）。
#include "nlohmann/json.hpp"

using WLJson = nlohmann::json;

// 把 workload 的所有 core-reference id 从「die 内 local」平移到 die_id 的 global id
// （id += die_id * CORES_PER_DIE），并置 id_space="global"。覆盖字段：
//   chips[*].cores[*].id / .prim_copy / .send_global_mem
//   chips[*].cores[*].worklist[*].cast[*].dest（整数、>=0 的核目的地）
//   source[*].dest
// 注：cast/recv 的 tag 语义分离留待后续，本函数只平移 core-reference。
// die_id==0 时为恒等（不改，兼容旧单 die 配置）。
void NormalizeWorkloadJson(WLJson &j, int die_id);

// 结构校验（bounds + 跨 die cast），只读、无副作用，可独立测试；会读取已构造的全局
// die/port/link 拓扑。非法抛 std::runtime_error。
// 不含「die>0 能否运行」的运行时判断（那属 HOST attachment 就绪与否，见 config_helper）。
//   - id_space 缺省=die0 local（旧配置兼容），"global"=[0,TOTAL_CORES)
//   - 越界 id、die0-local 下引用 die>0 报错；多跳/无实际双向 link 的跨 die cast 报错
//   - allow_d2d 默认 false，调用者需显式选择能力边界；V1-c3 生产 dataflow
//     路径传 true，放行「die 级维序路径上每一跳都存在精确双向 peer link」的 core cast
//     （V1 仅相邻；V2 起含多跳，故不再叫 allow_adjacent_d2d）。
void ValidateWorkloadStructure(const WLJson &j, int chip_id,
                               bool allow_d2d = false);
