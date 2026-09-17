#!/usr/bin/env python3
"""Configure, build, and check inside one managed frontend_tmp job.

Example:
  python3 scripts/frontend_tmp.py --root ./build-managed --max-gib 12 run \
    --name native-check -- python3 scripts/frontend_cmake.py \
    --source . --build-dir '{tmp}' --target npusim \
    --check '{tmp}/npusim' --isa-v1-selftest

The enclosing job removes a successful build immediately. Failed builds retain
only the configured frontend_tmp failure TTL; evidence jobs need explicit release.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


MARKER = '.waferai-frontend-root.json'
JOB_MARKER = '.waferai-frontend-job.json'


def managed_build_dir(path: Path) -> Path:
    path = path.resolve()
    env_path = os.environ.get('WAFERAI_TEMP_DIR')
    if env_path is None or Path(env_path).resolve() != path:
        raise ValueError('build-dir must equal the active managed job directory')
    if (not path.is_dir() or path.is_symlink()
            or not (path / JOB_MARKER).is_file()
            or not (path.parent / MARKER).is_file()):
        raise ValueError('build-dir is not a marked managed job')
    job = json.loads((path / JOB_MARKER).read_text())
    root = json.loads((path.parent / MARKER).read_text())
    if (job.get('name') != path.name or job.get('state') != 'running'
            or job.get('uid') != os.getuid() or root != {'format': 1, 'uid': os.getuid()}):
        raise ValueError('managed build job marker mismatch')
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.'))
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--target', action='append', required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--cmake-option', action='append', default=[])
    parser.add_argument('--check', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.jobs > 64:
        parser.error('--jobs must be 1..64')
    if any(not item or item.startswith('-') for item in args.target):
        parser.error('--target requires nonempty target names')
    try:
        build = managed_build_dir(args.build_dir)
        source = args.source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError('source must be a directory')
        subprocess.run(['cmake', '-S', str(source), '-B', str(build),
                        *args.cmake_option], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target',
                        *args.target, '-j', str(args.jobs)], check=True)
        check = args.check or []
        if check and check[0] == '--':
            check.pop(0)
        if check:
            subprocess.run(check, cwd=source, check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'frontend_cmake: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
