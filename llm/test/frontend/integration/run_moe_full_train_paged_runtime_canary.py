"""Audit real two-step MoE SGD paging and reject re-signed source drift.

This is an execution/timing gate only. Numerical training remains unproven.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resign(sidecar: dict) -> dict:
    result = copy.deepcopy(sidecar)
    result.pop('id', None)
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
    result['id'] = 'moe_full_train_paged_runtime_' + hashlib.sha256(encoded).hexdigest()[:20]
    return result


def run(npusim: Path, args: list[str], sidecar: Path, log: Path) -> tuple[int, str]:
    result = subprocess.run(
        [str(npusim), *args, '--moe-train-paged-runtime', str(sidecar)],
        cwd=npusim.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    log.write_text(result.stdout)
    return result.returncode, result.stdout


def check_positive(output: str, contract: dict) -> dict:
    for version in range(3):
        marker = f'[MOE_TRAIN_PAGED_STATE] version={version} bytes=952 '
        if output.count(marker) != 1:
            raise AssertionError(f'missing external authority version {version}')
    for step, reads, writes in ((0, 2432, 952), (1, 4864, 1904)):
        marker = (f'[MOE_TRAIN_PAGED_STEP] index={step} restore=46 writeback=19 '
                  f'route_loads=2 external_read_bytes={reads} '
                  f'external_write_bytes={writes} pending=0 pass=1')
        if marker not in output or f'[DENSE_SEQUENCE_PROGRAM_IO] index={step} probes=1 pass=1' not in output:
            raise AssertionError(f'physical step/ProgramIO {step} did not drain')
        if not re.search(rf'\[MOE_ALL_SGD_PARTIAL_SEQUENCE_STEP\] index={step} .*sgd=19 store=19 .*full_training=0 functional=0 pass=1', output):
            raise AssertionError(f'19 native SGD writes absent at step {step}')
    if ('[MOE_TRAIN_PAGED_INPUT] index=1 prior_writeback_completed=1 authority=external' not in output or
        '[MOE_TRAIN_PAGED_DMA_DRAIN] events=130 submitted=130 completed=130 external_read_bytes=4864 external_write_bytes=1904 hbm_read_bytes=1904 hbm_write_bytes=4864 pending=0 pass=1' not in output or
        output.count('[TRAIN_SGD] core=0') != 38 or
        '[DENSE_SEQUENCE_DRAIN] segments=2 one_shot=1' not in output):
        raise AssertionError('native train, external authority or final drain absent')
    lines = re.findall(
        r'\[MOE_TRAIN_PAGED_DMA_EVENT\] index=(\d+) step=(\d+) '
        r'linked_record=(\d+) kind=(\w+) state_ref=(\S+) bytes=(\d+) '
        r'issue_cycle=(\d+) completed_at_ticks=(\d+) '
        r'lsu_dependency_complete=1 pass=1', output)
    if len(lines) != 130:
        raise AssertionError(f'exactly 130 real LSU DMA events required, got {len(lines)}')
    for index, (row, event) in enumerate(zip(lines, contract['events'])):
        actual = tuple(row[:6])
        expected = (str(index), str(event['step_index']),
                    str(event['linked_record_index']), event['kind'],
                    event['state_ref'], str(event['size_bytes']))
        if actual != expected or int(row[7]) <= int(row[6]):
            raise AssertionError(f'physical signed event mismatch at {index}')
    cycles = re.findall(r'\[SIM_RESULT\] makespan_cycles=(\d+)', output)
    if len(cycles) != 1 or int(cycles[0]) <= 0:
        raise AssertionError('simulator cycle result absent')
    versions = re.findall(r'\[MOE_TRAIN_PAGED_STATE\] version=\d+ bytes=952 digest=([0-9a-f]{64})', output)
    return {'dma_events': 130, 'train_sgd': 38, 'makespan_cycles': int(cycles[0]),
            'external_read_bytes': 4864, 'external_write_bytes': 1904,
            'state_digests': versions, 'numeric_gradient_witness': False,
            'full_training_gate': 'closed'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--npusim', type=Path, required=True)
    parser.add_argument('--artifact-prefix', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--hardware-config', type=Path, required=True)
    parser.add_argument('--simulation-config', type=Path, required=True)
    parser.add_argument('--mapping-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(args.sidecar.read_text())
    assert resign(contract)['id'] == contract['id']
    prefix = str(args.artifact_prefix.resolve())
    native_args = [
        '--program-sequence', ','.join(f'{prefix}{i}.npup' for i in range(2)),
        '--linked-manifest-sequence', ','.join(f'{prefix}{i}.linked.json' for i in range(2)),
        '--program-io-sequence', ','.join(f'{prefix}{i}.program_io.json' for i in range(2)),
        '--moe-all-sgd-partial-sequence',
        '--hardware-config', str(args.hardware_config.resolve()),
        '--simulation-config', str(args.simulation_config.resolve()),
        '--mapping-config', str(args.mapping_config.resolve()),
        '--trace-window', '1000000',
    ]
    code, text = run(args.npusim.resolve(), native_args, args.sidecar.resolve(),
                     output / 'positive.npusim.log')
    if code:
        raise AssertionError(f'positive native MoE pager failed: {text[-2000:]}')
    receipt = check_positive(text, contract)
    mutants = {
        'linked_record': ('actual linked LSU order differs',
                          lambda j: j['events'][0].__setitem__('linked_record_index',
                                                                j['events'][0]['linked_record_index'] + 1)),
        'zero_seed': ('external seed omitted/duplicated',
                      lambda j: j['seeds'][0].__setitem__('payload_hex',
                                                         '00' * (len(j['seeds'][0]['payload_hex']) // 2))),
        'capacity': ('unsupported full two-step bounded MoE training contract',
                     lambda j: j.__setitem__('hbm_capacity_bytes', 2048)),
    }
    for name, (message, mutate) in mutants.items():
        changed = copy.deepcopy(contract)
        mutate(changed)
        changed = resign(changed)
        path = output / f'{name}.runtime.json'
        path.write_text(json.dumps(changed, sort_keys=True, separators=(',', ':')) + '\n')
        failure_code, failure_log = run(args.npusim.resolve(), native_args,
                                        path, output / f'{name}.npusim.log')
        if failure_code == 0 or message not in failure_log or '[MOE_TRAIN_PAGED_DMA_EVENT]' in failure_log:
            raise AssertionError(f'mutated {name} sidecar was not rejected before DMA')
    receipt.update({'native_sha256': digest(args.npusim.resolve()),
                    'sidecar_sha256': digest(args.sidecar.resolve()),
                    'positive_log_sha256': digest(output / 'positive.npusim.log'),
                    'negative_tests': list(mutants), 'status': 'pass'})
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
