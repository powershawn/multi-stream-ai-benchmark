import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark.config import Config
from benchmark.reporting import write_run_report


class RunnerTests(unittest.TestCase):
    def run_configured(self, counts, delay_ms):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(backend='synthetic', duration_s=1, warmup_s=0,
                            confirm_duration_s=1, confirm_repeats=1,
                            target_fps=5, window_s=1, test_streams=counts,
                            synthetic_delay_ms=delay_ms, output=directory)
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'benchmark', 'run',
                                     '--config', str(config_path)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary_path = next(Path(directory).glob('*/summary.json'))
            return json.loads(summary_path.read_text(encoding='utf-8'))

    def test_configured_cli_stops_after_first_failure(self):
        summary = self.run_configured([1, 2, 4], 300)
        self.assertEqual([(s['phase'], s['stream_count']) for s in summary['stages']],
                         [('search', 1)])
        self.assertEqual(summary['search_candidate_streams'], 0)
        self.assertIsNone(summary['max_stable_streams'])

    def test_configured_cli_confirms_only_highest_passing_listed_count(self):
        summary = self.run_configured([1, 3], 0)
        self.assertEqual([(s['phase'], s['stream_count']) for s in summary['stages']],
                         [('search', 1), ('search', 3), ('confirm', 3)])
        self.assertEqual(summary['search_candidate_streams'], 3)
        self.assertIsNone(summary['max_stable_streams'])

    def test_configured_report_does_not_claim_an_exact_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config().to_dict()
            config['test_streams'] = [1, 4, 8]
            write_run_report(Path(directory), dict(hardware_result=True, max_stable_streams=4,
                at_limit=False, confirmed=True, stages=[], config=config))
            report = (Path(directory) / 'report.md').read_text(encoding='utf-8')
            self.assertIn('configured test_streams', report)
            self.assertIn('not an absolute hardware maximum', report)
            self.assertNotIn('Maximum confirmed stable count in this search', report)

    def test_failed_confirmation_is_not_reported_as_no_short_test_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run_report(Path(directory), dict(hardware_result=True, max_stable_streams=0,
                at_limit=False, confirmed=True, search_candidate_streams=2,
                stages=[], config=Config().to_dict()))
            report = (Path(directory) / 'report.md').read_text(encoding='utf-8')
            self.assertIn('required confirmation', report)
            self.assertNotIn('No tested camera count passed', report)

    def test_synthetic_cli_cannot_claim_gpu_camera_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(backend='synthetic', duration_s=.4, warmup_s=0,
                            confirm_duration_s=.4, confirm_repeats=1,
                            target_fps=5, max_streams=1, window_s=.2,
                            synthetic_delay_ms=0, output=directory)
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
            command = [sys.executable, '-m', 'benchmark', 'run', '--config', str(config_path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summaries = list(Path(directory).glob('*/summary.json'))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding='utf-8'))
            self.assertFalse(summary['hardware_result'])
            self.assertIsNone(summary['max_stable_streams'])
            self.assertTrue((summaries[0].parent / 'summary.csv').is_file())
            self.assertTrue((summaries[0].parent / 'report.md').is_file())

    def test_report_labels_limit_as_lower_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run_report(Path(directory), dict(hardware_result=True, max_stable_streams=16,
                at_limit=True, confirmed=True, stages=[], config=Config().to_dict()))
            report = (Path(directory) / 'report.md').read_text(encoding='utf-8')
            self.assertIn('at least 16', report)
            self.assertIn('not an absolute hardware maximum', report)

    def test_aborted_search_report_does_not_describe_a_smoke_test(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run_report(Path(directory), dict(hardware_result=True, max_stable_streams=None,
                at_limit=False, confirmed=False, status='aborted', stages=[], config=Config().to_dict()))
            report = (Path(directory) / 'report.md').read_text(encoding='utf-8')
            self.assertIn('aborted', report)
            self.assertNotIn('Smoke test only', report)


if __name__ == '__main__':
    unittest.main()
