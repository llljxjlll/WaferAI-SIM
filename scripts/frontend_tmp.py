#!/usr/bin/env python3
"""Bounded, opt-in /tmp workspace for frontend builds and native experiments.

Only directories created by this program under its own marked root are pruned.
Native evidence is retained until explicitly released after review.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterator

DEFAULT_ROOT = Path('/tmp/waferai-frontend-managed')
ROOT_MARKER = '.waferai-frontend-root.json'
JOB_MARKER = '.waferai-frontend-job.json'
NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}\Z')
MONITOR_SECONDS = 5


def root_ready(root: Path) -> Path:
    if root.is_symlink() or root.resolve() in (Path('/'), Path('/tmp')):
        raise ValueError('unsafe managed tmp root')
    if not root.exists():
        root.mkdir(mode=0o700, parents=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('managed tmp root is not a real directory')
    marker = root / ROOT_MARKER
    if not marker.exists():
        if any(root.iterdir()):
            raise ValueError('nonempty unmanaged root cannot be adopted')
        marker.write_text(json.dumps({'format': 1, 'uid': os.getuid()}))
    elif marker.is_symlink() or json.loads(marker.read_text()) != {
        'format': 1, 'uid': os.getuid(),
    }:
        raise ValueError('managed tmp root marker mismatch')
    return root.resolve()


@contextmanager
def locked(root: Path) -> Iterator[None]:
    with (root / '.lock').open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def metadata(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_dir() or not path.name.startswith('job-'):
        return None
    marker = path / JOB_MARKER
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        info = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if (info.get('format') != 1 or info.get('name') != path.name
            or info.get('uid') != os.getuid()
            or info.get('kind') not in ('scratch', 'evidence')
            or info.get('state') not in ('running', 'failed', 'held', 'released')):
        return None
    return info


def save(path: Path, info: dict[str, object]) -> None:
    temp = path / (JOB_MARKER + '.new')
    temp.write_text(json.dumps(info, sort_keys=True) + '\n')
    temp.replace(path / JOB_MARKER)


def pid_start(pid: int) -> str | None:
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def boot_id() -> str:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def group_alive(info: dict[str, object]) -> bool:
    group = info.get('process_group')
    if (info.get('boot_id') != boot_id() or type(group) is not int
            or group <= 0):
        return False
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def running(info: dict[str, object]) -> bool:
    pid = info.get('pid')
    return (type(pid) is int and pid > 0
            and info.get('pid_start') is not None
            and info['pid_start'] == pid_start(pid)) or group_alive(info)


def size_bytes(path: Path) -> int:
    total = 0
    for folder, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(folder) / name).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def candidates(root: Path, now: float, failure_ttl: float, *, apply: bool) -> list[tuple[Path, dict[str, object]]]:
    result = []
    for path in root.iterdir():
        info = metadata(path)
        if info is None or group_alive(info):
            continue
        if info['state'] == 'running':
            if running(info):
                continue
            info = dict(info, state='failed', finished=now)
            if apply:
                save(path, info)
        if info['state'] == 'released' or (info['state'] == 'failed'
                and now - float(info.get('finished', now)) >= failure_ttl):
            result.append((path, info))
    return sorted(result, key=lambda pair: float(pair[1].get('finished', 0)))


def prune(root: Path, *, apply: bool, max_bytes: int, failure_ttl: float) -> dict[str, object]:
    with locked(root):
        now = time.time()
        victims = candidates(root, now, failure_ttl, apply=apply)
        size = size_bytes(root)
        # Expired scratch and released evidence are eligible. Running or held
        # evidence never becomes a pressure victim, even above the budget.
        freed = sum(size_bytes(path) for path, _ in victims)
        if apply:
            for path, info in victims:
                if metadata(path) != info:
                    raise ValueError(f'job marker drifted: {path}')
                shutil.rmtree(path)
        return {'root': str(root), 'bytes_before': size,
                'bytes_after': size - freed if apply else size,
                'budget_bytes': max_bytes, 'applied': apply,
                'removed': [str(path) for path, _ in victims],
                'over_budget': size - (freed if apply else 0) > max_bytes}


def execute(args: argparse.Namespace, root: Path, *,
            resume_path: Path | None = None) -> int:
    if resume_path is None and not NAME.fullmatch(args.name):
        raise ValueError('job name must be 1..40 ASCII letters, numbers, _ or -')
    command = list(args.command)
    if command and command[0] == '--':
        command.pop(0)
    if not command:
        raise ValueError('run requires an executable after --')
    dramsys_root = None
    if getattr(args, 'dramsys_root', None) is not None:
        dramsys_root = args.dramsys_root.resolve(strict=True)
        if not dramsys_root.is_dir():
            raise ValueError('DRAMSys root must be a directory')
    if resume_path is None:
        prune(root, apply=True, max_bytes=args.max_bytes,
              failure_ttl=args.failure_ttl)
    with locked(root):
        if size_bytes(root) >= args.max_bytes:
            raise ValueError('managed tmp budget exhausted; release evidence or raise explicit budget')
        if resume_path is None:
            path = Path(tempfile.mkdtemp(prefix='job-' + args.name + '-', dir=root))
            info: dict[str, object] = {
                'format': 1, 'name': path.name, 'uid': os.getuid(),
                'kind': args.kind, 'state': 'running', 'created': time.time(),
                'pid': os.getpid(), 'pid_start': pid_start(os.getpid()),
                'process_group': None, 'boot_id': boot_id(),
            }
            save(path, info)
            if dramsys_root is not None:
                # Native NpuSim resolves ../DRAMSys/configs from a nested Fresh cwd.
                # A symlink keeps immutable configuration outside the quota.
                (path / 'DRAMSys').symlink_to(
                    dramsys_root, target_is_directory=True)
        else:
            path = resume_path.absolute()
            if path.parent.resolve() != root or path.is_symlink():
                raise ValueError('resume only accepts a direct managed job directory')
            prior = metadata(path)
            if (prior is None or prior['state'] not in ('failed', 'running')
                    or running(prior)):
                raise ValueError('resume requires one stopped marked job')
            info = dict(prior, state='running', resumed=time.time(),
                        pid=os.getpid(), pid_start=pid_start(os.getpid()),
                        process_group=None, boot_id=boot_id())
            save(path, info)
    env = dict(os.environ, TMPDIR=str(path), WAFERAI_TEMP_DIR=str(path))
    command = [part.replace('{tmp}', str(path)) for part in command]
    print(json.dumps({'job': str(path), 'command': command}), flush=True)
    process = None
    code = 1
    termination_signal: int | None = None

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal termination_signal
        termination_signal = signum
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        with locked(root):
            info['pid'] = process.pid
            info['pid_start'] = pid_start(process.pid)
            info['process_group'] = process.pid
            save(path, info)
        while True:
            try:
                code = process.wait(timeout=MONITOR_SECONDS)
                break
            except subprocess.TimeoutExpired:
                # Reclaim expired failures while long native jobs keep running.
                state = prune(root, apply=True, max_bytes=args.max_bytes,
                              failure_ttl=args.failure_ttl)
                if state['over_budget']:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError('managed tmp root exceeded byte budget')
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        code = 128 + (termination_signal or signal.SIGINT)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if process is not None and group_alive(info):
            # The direct command may exit while a background writer keeps its
            # process group alive. Never leave that writer filling this job.
            if code == 0:
                code = 2
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.1)
            if group_alive(info):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if code == 0 and size_bytes(root) > args.max_bytes:
            code = 2
        with locked(root):
            info['state'] = ('held' if code == 0 and info['kind'] == 'evidence'
                             else 'released' if code == 0 else 'failed')
            info['finished'] = time.time()
            info['pid'] = None
            info['pid_start'] = None
            save(path, info)
        prune(root, apply=True, max_bytes=args.max_bytes,
              failure_ttl=args.failure_ttl)
    return code


def release(root: Path, path: Path, *, max_bytes: int, failure_ttl: float) -> dict[str, object]:
    if path.parent.resolve() != root or path.is_symlink():
        raise ValueError('release only accepts a direct managed job directory')
    with locked(root):
        info = metadata(path)
        if info is None or running(info):
            raise ValueError('job is unmarked or still running')
        info['state'] = 'released'
        info['finished'] = time.time()
        save(path, info)
    return prune(root, apply=True, max_bytes=max_bytes, failure_ttl=failure_ttl)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--max-gib', type=float, default=8)
    parser.add_argument('--failure-hours', type=float, default=24)
    sub = parser.add_subparsers(dest='operation', required=True)
    run = sub.add_parser('run', help='run a command with bounded TMPDIR and {tmp} substitution')
    run.add_argument('--name', required=True)
    run.add_argument('--kind', choices=('scratch', 'evidence'), default='scratch')
    run.add_argument('--dramsys-root', type=Path,
                     help='link immutable DRAMSys configs for nested native Fresh cwd')
    run.add_argument('command', nargs=argparse.REMAINDER)
    resume = sub.add_parser('resume', help='rerun in a stopped marked job without copying its evidence')
    resume.add_argument('job', type=Path)
    resume.add_argument('command', nargs=argparse.REMAINDER)
    sub.add_parser('prune', help='dry-run by default; only remove managed jobs')
    sub.choices['prune'].add_argument('--apply', action='store_true')
    done = sub.add_parser('release', help='remove a reviewed evidence job')
    done.add_argument('job', type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.max_gib < 1024 or not 0 <= args.failure_hours <= 24 * 365:
        parser.error('invalid managed tmp budget or failure retention')
    args.max_bytes = int(args.max_gib * 1024 ** 3)
    args.failure_ttl = args.failure_hours * 3600
    try:
        root = root_ready(args.root)
        if args.operation == 'run':
            return execute(args, root)
        if args.operation == 'resume':
            return execute(args, root, resume_path=args.job)
        if args.operation == 'release':
            result = release(root, args.job.absolute(),
                             max_bytes=args.max_bytes, failure_ttl=args.failure_ttl)
        else:
            result = prune(root, apply=args.apply, max_bytes=args.max_bytes,
                           failure_ttl=args.failure_ttl)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'frontend_tmp: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
