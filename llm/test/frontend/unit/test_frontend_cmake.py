"""A managed CMake job must build, check, and release its build tree."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
TEMP_SCRIPT = ROOT / 'scripts/frontend_tmp.py'
CMAKE_SCRIPT = ROOT / 'scripts/frontend_cmake.py'


class ManagedCmakeTest(unittest.TestCase):
    def test_build_check_and_cleanup_in_one_job(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            source = parent / 'source'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.16)\n'
                'project(managed_build LANGUAGES CXX)\n'
                'add_executable(gate gate.cpp)\n', encoding='utf-8')
            (source / 'gate.cpp').write_text(
                '#include <iostream>\n'
                'int main() { std::cout << "GATE_OK\\n"; }\n', encoding='utf-8')
            managed = parent / 'build-managed'
            command = [sys.executable, str(TEMP_SCRIPT), '--root', str(managed),
                       '--max-gib', '0.1', '--failure-hours', '0', 'run',
                       '--name', 'compile-gate', '--', sys.executable,
                       str(CMAKE_SCRIPT), '--source', str(source),
                       '--build-dir', '{tmp}', '--target', 'gate', '--jobs', '2',
                       '--check', '{tmp}/gate']
            result = subprocess.run(command, text=True, capture_output=True,
                                    check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('GATE_OK', result.stdout)
            job = Path(json.loads(result.stdout.splitlines()[0])['job'])
            self.assertFalse(job.exists())
            self.assertEqual(list(managed.glob('job-*')), [])
            command[command.index('gate', command.index('--target') + 1)] = 'absent'
            failed = subprocess.run(command, text=True, capture_output=True,
                                    check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(list(managed.glob('job-*')), [])

    def test_refuses_unmanaged_build_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = subprocess.run(
                [sys.executable, str(CMAKE_SCRIPT), '--source', str(path),
                 '--build-dir', str(path / 'build'), '--target', 'gate'],
                text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((path / 'build').exists())


if __name__ == '__main__':
    unittest.main()
