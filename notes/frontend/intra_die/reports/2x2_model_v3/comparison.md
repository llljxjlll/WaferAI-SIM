# Intra-die 2x2 ablation

Scope: `dense_tp2_gemm_rs_common_ir2`; status: `fail_uncalibrated`; calibrated: `false`.

| Case | inter_die | intra_die | cycles | projection digest |
|---|---|---|---:|---|
| A00 | naive | off | 14760 | `691be063013c6b30ba7ce1d2a2781ecb5c7226735ef02acd3d276af14ef77c6d` |
| A10 | swizzle_topo | off | 6391 | `fb737b8a5da23bbc8013d6bf4f663a5e493bd121c2aec9d3b6d0013fcd678171` |
| A01 | naive | auto | 14760 | `691be063013c6b30ba7ce1d2a2781ecb5c7226735ef02acd3d276af14ef77c6d` |
| A11 | swizzle_topo | auto | 6391 | `fb737b8a5da23bbc8013d6bf4f663a5e493bd121c2aec9d3b6d0013fcd678171` |

## Comparison (interaction = A11 - A10 - A01 + A00 cycles)

- intra_speedup_naive_inter: `1`
- intra_speedup_swizzle_inter: `1`
- inter_speedup_naive_intra: `2.30949773`
- inter_speedup_optimized_intra: `2.30949773`
- combined_speedup: `2.30949773`
- interaction: `0`
