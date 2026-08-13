# NPUSim Program Format v1

本文档描述 `--program` 使用的稳定外部容器。它不是内部 `PrimBase::serialize()` 产生的 128-bit wire；loader 必须先解析本格式中的外部 ISA record，完成符号重定位和 capability 校验，再 lowering 成内部 Prim wire。

## 1. 基本规则

- 所有整数使用 little-endian；地址和长度单位均为 byte，除非对应 opcode schema 另有说明。
- format version=`1.0`，ISA version=`1.0`；不支持的 major/minor、endianness、required section 或 capability 必须在下发任何 CONFIG 前拒绝。
- 文件上限 64 MiB；字符串必须是无 NUL 的规范 UTF-8，单项最长 255 bytes。
- encoder 产生 7 个 required sections，按 type 1 到 7 排列且无 gap。decoder 可跳过未知 optional section，但拒绝未知 required section。
- 每个 section 和整个文件使用 CRC-32C (Castagnoli，初值/终值 XOR `0xffffffff`)。计算 whole-file CRC 时将 header 的 `[56,60)` 当作 0。

## 2. 64-byte header

| Offset | Size | Field | v1 value/meaning |
|---:|---:|---|---|
| 0 | 8 | magic | ASCII `NPUPRG1` 加结尾 `00` |
| 8 | 2 | format_major | `1` |
| 10 | 2 | format_minor | `0` |
| 12 | 2 | isa_major | `1` |
| 14 | 2 | isa_minor | `0` |
| 16 | 2 | header_size | `64` |
| 18 | 1 | endianness | `1`，little-endian |
| 19 | 5 | reserved | 0 |
| 24 | 8 | capabilities | 外部 opcode capability bitmap |
| 32 | 4 | section_count | v1 canonical encoder 为 7，decoder 上限 64 |
| 36 | 4 | reserved | 0 |
| 40 | 8 | section_table_offset | canonical 为 64 |
| 48 | 8 | file_size | 必须等于实际输入边界 |
| 56 | 4 | whole_file_crc32c | whole-file CRC-32C |
| 60 | 4 | reserved | 0 |

每个 section descriptor 固定 40 bytes：`type:u32, flags:u32, offset:u64, size:u64, count:u32, entry_size:u32, crc32c:u32, reserved:u32`。v1 已知 section 的 flags 必须为 `REQUIRED=1`。

## 3. Sections

| Type | Name | Encoding |
|---:|---|---|
| 1 | STRING_TABLE | 重复 `length:u32 + UTF-8 bytes`；不允许重复字符串 |
| 2 | SYMBOL_TABLE | 固定 24 bytes/项：`name_string_index:u32, kind:u8, flags:u8, reserved:u16, value:u64, size_bytes:u64` |
| 3 | SEMANTIC_RELOCATION_TABLE | 固定 24 bytes/项：`core_index:u32, instruction_index:u32, operand_id:u16, kind:u8, reserved:u8, symbol_index:u32, addend:i64` |
| 4 | CORE_GROUP_TABLE | 重复 `group_id:u32, member_count:u32, members:u16[]` |
| 5 | CORE_PROGRAM_INDEX | 固定 32 bytes/项：`core_id:u16, flags:u16, record_count:u32, stream_offset:u64, stream_size:u64, reserved:u64` |
| 6 | EXTERNAL_RECORD_STREAM | 各 core 的外部 record 连续串；边界和 record codec 由 `record_codec.h` 定义 |
| 7 | CONTROL_ENVELOPE | source/start、terminal、ACK/DONE 集合及空 core 策略 |

symbol kind：`1=ABSOLUTE_ADDRESS`、`2=SRAM_REGION`、`3=SRAM_LABEL`。artifact 保存稳定字符串/符号索引，禁止保存进程内 `g_addr_label_table` 数字 ID。

`SRAM_REGION` 的 `value` 唯一表示该 region 的物理 byte base，`size_bytes` 表示从该 base 起的有界 byte extent。引用 `SRAM_REGION` 的 relocation 其 `addend` 唯一表示 region 内 byte offset；loader 保留 named-region 操作数时不得把 `value` 加入 offset，转成绝对地址时必须且只能计算一次 `value + addend`。负 addend、base/offset/span 溢出以及 `offset + access_bytes > size_bytes` 均在任何 lowering 派生状态或运行时写入前拒绝。SCATTER source、GATHER/REDUCE destination 与 reduction staging 的 `access_bytes=N*L`，普通 endpoint 和 reduction result 的 `access_bytes=L`。

core/group/core-list 必须严格递增且唯一。v1 loader 还会结合已加载平台检查 core 范围、active membership 以及 group 同 die 约束。

## 4. Control envelope

header 固定 32 bytes：

`version:u16, empty_core_ack_policy:u8, failure_policy:u8, active_count:u32, start_count:u32, terminal_count:u32, ack_count:u32, done_count:u32, reserved:u64`

随后依次编码：

1. `active_cores:u16[]`；
2. start event，每项 12 bytes：`target_core:u16, reserved:u16, tag:u32, count:u32`；
3. `terminal_cores:u16[]`；
4. `expected_ack_cores:u16[]`；
5. `expected_done_cores:u16[]`。

`INCLUDE_EMPTY` 时 ACK 集合必须等于全部 program cores；`EXCLUDE_EMPTY` 时只含 record stream 非空的 cores。DONE 集合必须恰好等于 terminal 集合。当前 host wire 进一步限制 start `tag<=65535`、`1<=count<=255`，超限由 program helper 确定性拒绝，不截断。

## 5. 加载和兼容顺序

loader 采用两阶段提交：

1. 校验文件边界、版本、sections、CRC、record、引用、capability 和控制 envelope；
2. 在临时 artifact 上应用 semantic relocation；
3. 校验 platform core/group；
4. lowering 外部 opcode，并对全部内部 Prim wire 做 serialize 验证；
5. 原子替换 `coreconfigs`、ACK/DONE 状态和每核队列。

任一步失败都不得向 core 下发部分 CONFIG。v1 不接受大端 artifact，不对未知版本做猜测性兼容。后续不兼容字段必须提升 format 或 ISA version；新增 section 只有在旧 loader 可安全忽略时才可标记 optional。

## 6. 参考编码器

构建后运行：

```bash
build/npusim_program_fixture "build/program fixture 路径.npup"
build/npusim --program "build/program fixture 路径.npup"
```

参考工具生成一个 core 0 的最小可终止 artifact，覆盖空指令流、CONFIG ACK、START 和 DONE。它仅链接外部 ISA/container codec，不初始化 SystemC 或全局运行时，可作为编译器后端的最小编码示例。
