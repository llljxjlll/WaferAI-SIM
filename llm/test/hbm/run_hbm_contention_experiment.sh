#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
npusim="${1:-${repo_root}/build/npusim}"
pairs="${HBM_EXPERIMENT_PAIRS:-8}"
bytes="${HBM_EXPERIMENT_BYTES:-1024}"

if [[ ! -x "${npusim}" ]]; then
    echo "npusim executable not found: ${npusim}" >&2
    exit 2
fi

bin_dir="$(cd "$(dirname "${npusim}")" && pwd)"
bin_name="$(basename "${npusim}")"

echo "port,mem_tile,cores,pairs_per_core,bytes_per_access,logical_requests,reads,writes,elapsed_ns,useful_GBps,mean_latency_ns,p95_latency_ns,max_latency_ns,noc_hops,request_flits,response_flits,injection_stalls,mem_mem_noc_contention,endpoint_queue_stalls,endpoint_queue_wait_ns,backend_service_ns,network_residual,router_residual,data_errors"
for port in N0 N3; do
    for cores in 1 2 4 8; do
        result="$(
            cd "${bin_dir}"
            "./${bin_name}" --hbm-contention-experiment                 --hbm-experiment-port="${port}"                 --hbm-experiment-cores="${cores}"                 --hbm-experiment-pairs="${pairs}"                 --hbm-experiment-bytes="${bytes}" |
                sed -n '/^HBM_EXPERIMENT_CSV,/p'
        )"
        if [[ -z "${result}" ]]; then
            echo "missing experiment result for ${port}/${cores} cores" >&2
            exit 3
        fi
        echo "${result#HBM_EXPERIMENT_CSV,}"
    done
done
