# C0：配置与兼容契约

日期：2026-08-25  
状态：完成

## 实现

- 增加 `legacy_shared`、`dual_dte_dedicated` 模式。
- 增加 queue depth、dispatch width、dispatch latency、notify latency 四项硬件参数。
- 支持顶层默认、逐 core merge 覆盖、core gap/tail 自动继承和多 die local-core 复用。
- 单 core 从全局 dedicated 切回 legacy 时恢复 dedicated 资源默认值。
- 严格拒绝未知模式、零深度/宽度、width 大于 depth、负数和不可安全转换的延迟。
- 默认硬件显式记录 `legacy_shared`；启动日志按同构 core 范围汇总有效配置。

## 证据

- 配置 selftest 覆盖缺省、全局 dedicated、自动补齐、逐核 legacy 和非法参数。
- `control_cores` 缺失时保持 legacy 默认，不改变 `TOTAL_CORES`、endpoint 或 placement。

