# Intra-die 2x2 ablation

Scope: `dense_tp2_gemm_rs_common_ir2`; calibrated: `false`.

| Case | inter_die | intra_die | cycles | projection digest |
|---|---|---|---:|---|
| A00 | naive | naive | 8964 | `3538d6dc45f6c0bb69e7f508c51653a19f5fbe254c73eb7ebeec37d3edf60022` |
| A10 | swizzle_topo | naive | 2721 | `a5b8985821e8855e36f3fa711654ff4d14b739a23d8d598655c926d52aa0fc67` |
| A01 | naive | optimized | 8964 | `3538d6dc45f6c0bb69e7f508c51653a19f5fbe254c73eb7ebeec37d3edf60022` |
| A11 | swizzle_topo | optimized | 2721 | `a5b8985821e8855e36f3fa711654ff4d14b739a23d8d598655c926d52aa0fc67` |

## Comparison (interaction = A11 - A10 - A01 + A00 cycles)

- intra_speedup_naive_inter: `1`
- intra_speedup_swizzle_inter: `1`
- inter_speedup_naive_intra: `3.29437707`
- inter_speedup_optimized_intra: `3.29437707`
- combined_speedup: `3.29437707`
- interaction: `0`
