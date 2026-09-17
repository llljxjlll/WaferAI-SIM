"""The opt-in temp janitor must not remove live or unmarked evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[4] / 'scripts/frontend_tmp.py'


class ManagedFrontendTmpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / 'waferai-frontend-managed'
        self.base = [sys.executable, str(SCRIPT), '--root', str(self.root),
                     '--max-gib', '0.01', '--failure-hours', '0']

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([*self.base, *args], text=True,
                              capture_output=True, check=False)

    def test_successful_scratch_cleans_and_preserves_unmarked_sibling(self) -> None:
        untouched = self.parent / 'unmanaged-evidence'
        untouched.mkdir()
        (untouched / 'proof').write_text('keep')
        result = self.run_cli('run', '--name', 'trial', '--', sys.executable,
                              '-c', "from pathlib import Path; import os; Path(os.environ['WAFERAI_TEMP_DIR'], 'scratch').write_text('worked')")
        self.assertEqual(result.returncode, 0, result.stderr)
        job = Path(json.loads(result.stdout.splitlines()[0])['job'])
        self.assertFalse(job.exists())
        self.assertEqual((untouched / 'proof').read_text(), 'keep')

    def test_failed_scratch_ttl_and_evidence_release(self) -> None:
        self.base[-1] = '1'
        failed = self.run_cli('run', '--name', 'failure', '--', sys.executable,
                              '-c', 'raise SystemExit(3)')
        self.assertEqual(failed.returncode, 3)
        failed_job = Path(json.loads(failed.stdout.splitlines()[0])['job'])
        self.assertTrue(failed_job.exists())
        original_marker = (failed_job / '.waferai-frontend-job.json').read_bytes()
        self.base[-1] = '0'
        preview = self.run_cli('prune')
        self.assertIn(str(failed_job), json.loads(preview.stdout)['removed'])
        self.assertEqual((failed_job / '.waferai-frontend-job.json').read_bytes(), original_marker)
        self.assertTrue(failed_job.exists())
        self.assertEqual(self.run_cli('prune', '--apply').returncode, 0)
        self.assertFalse(failed_job.exists())
        evidence = self.run_cli('run', '--kind', 'evidence', '--name', 'native',
                                '--', sys.executable, '-c',
                                "from pathlib import Path; import os; Path(os.environ['WAFERAI_TEMP_DIR'], 'proof').write_text('kept')")
        self.assertEqual(evidence.returncode, 0, evidence.stderr)
        job = Path(json.loads(evidence.stdout.splitlines()[0])['job'])
        self.assertEqual((job / 'proof').read_text(), 'kept')
        self.assertEqual(json.loads(self.run_cli('prune', '--apply').stdout)['removed'], [])
        self.assertTrue(job.is_dir())
        released = self.run_cli('release', str(job))
        self.assertEqual(released.returncode, 0, released.stderr)
        self.assertFalse(job.exists())

    def test_running_and_unmarked_jobs_are_not_pruned(self) -> None:
        initialized = self.run_cli('prune')
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        unmarked = self.root / 'job-external'
        unmarked.mkdir()
        job = self.root / 'job-real'
        job.mkdir()
        marker = job / '.waferai-frontend-job.json'
        marker.write_text(json.dumps({'format': 1, 'uid': os.getuid(),
                                      'name': job.name, 'kind': 'scratch',
                                      'state': 'running', 'pid': os.getpid(),
                                      'pid_start': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]}))
        result = self.run_cli('prune', '--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(job.exists())
        self.assertTrue(unmarked.exists())
        self.assertNotEqual(self.run_cli('release', str(job)).returncode, 0)

    def test_native_job_links_dramsys_within_managed_root(self) -> None:
        config = self.parent / 'frozen-dramsys'
        (config / 'configs').mkdir(parents=True)
        (config / 'configs' / 'hbm.json').write_text('{}')
        result = self.run_cli(
            'run', '--name', 'native', '--dramsys-root', str(config), '--',
            sys.executable, '-c',
            "from pathlib import Path; import os; p=Path(os.environ['TMPDIR']); "
            "assert (p/'DRAMSys/configs/hbm.json').read_text() == '{}'; "
            "print('DRAM_LINK_OK')",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('DRAM_LINK_OK', result.stdout)
        self.assertFalse(Path(json.loads(result.stdout.splitlines()[0])['job']).exists())
        self.assertEqual((config / 'configs' / 'hbm.json').read_text(), '{}')

    def test_job_exceeding_budget_is_not_published_as_success(self) -> None:
        self.base[5] = '0.000001'
        result = self.run_cli('run', '--name', 'oversize', '--kind', 'evidence',
                              '--', sys.executable, '-c',
                              "from pathlib import Path; import os; Path(os.environ['TMPDIR'], 'big').write_bytes(b'x' * 4096)")
        self.assertEqual(result.returncode, 2, result.stderr)
        path = Path(json.loads(result.stdout.splitlines()[0])['job'])
        self.assertFalse(path.exists())

    def test_refuses_nonempty_unmarked_root_and_path_traversal(self) -> None:
        self.root.mkdir()
        (self.root / 'foreign').write_text('keep')
        self.assertNotEqual(self.run_cli('prune', '--apply').returncode, 0)
        self.assertEqual((self.root / 'foreign').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
