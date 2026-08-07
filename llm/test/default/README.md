# Default smoke workload

This directory is the canonical no-argument `npusim` configuration bundle.
It intentionally uses one active core and the legacy automatic memory path so
it remains a fast compatibility baseline when the SRAM extension is disabled.

The workload must complete with one host `S_DATA` flow and one `DONE`
message. Changes to any file in this directory must keep the
`default_workload_smoke` CTest passing.
