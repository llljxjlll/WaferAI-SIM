# exp2-1 调试记录

## 已修复

1. **目标时钟误写为 1 GHz**  
   目标仿真周期是 2 ns。解析器已改为 500 MHz，并新增 cycles↔seconds、tokens/s 公式测试。
   周期数相应变化，物理秒数与吞吐保持按 rate/s 推导的一致口径。

2. **MLA 分支丢失 evidence signature**  
   DeepSeek 的 `analytical_only_mla` 提前返回曾清空 Stage3/Stage4/exp1 结构证据哈希。
   现先绑定 provenance，再选择 MLA estimate source。

3. **容量字段覆盖 E2E limitation tags**  
   `{**estimate, **capacity}` 曾使 vector/collective/NoC/HBM 抽象限制消失。现容量字段与模型
   限制集合合并；prefill-only 按自己的容量状态重算，不再继承 B64 decode 的 projection tag。

4. **GELU/SwiGLU 混算**  
   GPT-3 dense GELU 已改为两矩阵、4SHI；SwiGLU 保持三矩阵、6SHI。

5. **DeepSeek dense prefix 与 shared expert**  
   前三层固定 dense I=18432。主路径按方案只计算 routed top-8；shared expert 从主柱移出，
   使用 8 个独立 sensitivity case。

6. **MoE skew 破坏总工作量守恒**  
   旧实现直接把 tensor/vector/collective 总 work 乘 λ。现 logical work、bytes 与 HBM work
   对所有 λ 保持相同，只放大最忙 expert/route 的 runtime service；训练 forward/backward/WGRAD
   和推理 prefill/decode 一致处理，optimizer 不受 λ 影响。

7. **MoE weight service 只计算 top-k 权重**  
   现按 `min(E, tokens×top_k)` 计算实际触达专家，并在训练中将 routed expert 权重按 EP=4
   分片，non-routed 权重复制。

8. **结果 digest 自引用**  
   λ=1 通过复制主记录生成，旧 `result_digest` 曾进入新 digest。现 finalize 前移除旧 digest。

9. **DeepSeek 双重图形限制被单纹理隐藏**  
   SVG 新增 MLA+capacity 组合纹理；可行 prefill、capacity projection 与 MLA-only 来源可区分。

10. **训练仅前向 overlap 导致端到端收益被稀释**  
    新增 `full_train_overlap` 第三态。下一层 dX 不再被上一层 WGRAD/sync 串行阻塞，
    WGRAD 与逐层 gradient collective 可以流水；每层 sync 的 WGRAD 完成门禁以及 AdamW
    的全梯度门禁仍保留。三态 action 工作量完全一致，只改变依赖。12/12 个主训练点相对
    base 的 forward/backward/WGRAD 都严格变短，full-train 总加速均值从 forward-only 的
    1.0190× 提升到 1.0604×。

11. **Decode 把 KV/HBM 流量误算成 reducer 工作，且固定切成 8 个窗口**  
    旧实现将累计 KV read 加到 `activation_bytes`，使 `NORM_RESIDUAL` 错误处理远大于
    `B×H×dtype` 的字节，并允许 B64 获得不存在的 8-way chunk 独立性，因此约 1.48× 的
    “重叠收益”主要是伪造的 HBM/reducer 流水。v4 将 KV read 单独放入 HBM boundary stream，
    新 KV 显式 append 回 HBM；B64/B512 使用 1/4 个窗口，并按每窗口 M/128 计算利用率。
    collective 必须等待本窗口 HBM 数据 ready，再支付 2-cycle DTE launch 和 1 cycle/hop；
    HBM 支付 10-cycle first-byte。两 token commit 链等待全部 layer append，但只解释为固定
    KV 上下文附近的 local interval，未声称稳态收敛。修正后 B64 六个模型均为 1.0000×，
    B512 为 1.0003×–1.0504×；固定延迟、shape efficiency 和 KV append 仍是带 limitation
    tag 的解析先验。

## 未解决但已强制降级

- D2D parser 的单 lane 上限是 1 packet/cycle；16 B payload、2 ns/cycle 只能达到 8 GB/s，
  无法表达目标 1 TB/s。差距 125×。
- 没有目标绑定的 isolated GEMM、vector、NoC/DTE/SRAM/HBM saturation sweep。
- Stage3 当前 artifact digest 与 reviewed golden 不一致。
- DeepSeek MLA 没有可用的周期精确 capability。
- E2E 仍采用 representative collective route、die-level local NoC 与 HBM ingress 抽象。
- routing skew 使用 λ 合成最忙资源比例，没有真实 router assignment trace。

因此任何一条结果都不能升级为 `cycle_accurate_direct`。后续若要升级，优先顺序是：
修复 D2D 速率表达、完成目标 unit closure、补目标 primitive sweeps、生成至少四对直接
counterfactual E2E 校验，再以 p95 relative error ≤15% 作为发布门禁。
