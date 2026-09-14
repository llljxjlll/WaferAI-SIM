# Rectangular Mesh End-to-End Acceptance Evidence

This directory is the small, reproducible experiment batch for the current
rectangular Die mesh workload work. It invokes existing frontend production tests
and canaries; it does not implement another compiler or simulator path.

The classifications in `configs/cases.json` are deliberately separate:

- `preflight_pass` covers the 400-case `4 families × 100 shapes` schema,
  placement, graph, transport, and memory preflight matrix. It does not run the
  finalizer, resolver, or NpuSim.
- `runtime_pass` is limited to the real 1×1 Dense inference and two-step SGD
  canaries currently available, including their bounded-HBM offload variants.
- `compile_only` covers the 1×2/EP2 MoE compile evidence. The full-model MoE
  composite is not linked into one executable runtime program.
- `known_blocker` records a diagnostic that must fail with its declared message.
  The current 2×2 Dense runtime reaches the production P2P receive-session
  capacity gate and is excluded from the default case set.

These results do not complete M1, M2, or M3 from the opt_2609 plan. In
particular, the repository does not yet have four-family full-model runtime
coverage, the 800-case resident/offload P5 matrix with 1,600 independent
executions, AdamW coverage, or a successful shape beyond 10×10.

## Files

- `configs/cases.json` freezes the case IDs, evidence scopes, allowlisted Python
  module entrypoints, and expected output markers.
- `run_acceptance.py` records the source commit/tree binding, tool and config
  SHA-256 values, command vectors, return codes, wall time, and stdout/stderr
  digests. It reports counts by evidence outcome and never emits one aggregate
  `SUCCESS` classification.
- `tests/test_run_acceptance.py` checks config closure, the entrypoint allowlist,
  command construction, and fail-closed outcome classification without running
  a long experiment.

The runner writes raw logs and `run_manifest.json` only to the explicitly chosen
output directory. Simulator artifacts use separate, case-specific paths directly
under `/tmp`. They are frozen in the case config because the current
NpuSim/DRAMSys configuration stack has shown path-dependent behavior for nested
artifact directories. The manifest records each path and the exact command.
Neither directory is part of the committed result batch.

## Inspect and validate

```bash
python3 -B exps/rect_mesh_e2e/run_acceptance.py --list
python3 -B exps/rect_mesh_e2e/run_acceptance.py --dry-run
python3 -B -m unittest exps.rect_mesh_e2e.tests.test_run_acceptance -v
```

## Run the default bounded evidence set

Build the existing tools first:

```bash
cmake --build build-debug-final \
  --target npusim_program_finalizer npusim \
  npusim_external_dma_program_selftest -j2
```

Then run the default cases. The known 2×2 blocker is not included:

```bash
python3 -B exps/rect_mesh_e2e/run_acceptance.py \
  --output /tmp/rect_mesh_e2e_acceptance
```

Select one or more cases with repeated `--case` arguments. Reproducing the 2×2
diagnostic is explicit because its production compile is long:

```bash
python3 -B exps/rect_mesh_e2e/run_acceptance.py \
  --case dense_inference_runtime_2x2_session_blocker \
  --output /tmp/rect_mesh_e2e_2x2_blocker
```

The full 400-case preflight test previously took about 373 seconds. The 2×2
compile previously took about 845 seconds before reaching its runtime blocker.
Runtime canaries retain a per-case default timeout of 600 seconds; actual wall
times are recorded rather than inferred from simulator cycles.
