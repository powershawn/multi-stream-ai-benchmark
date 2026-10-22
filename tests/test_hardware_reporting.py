import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.config import Config
from benchmark.reporting import write_json, write_run_report
from benchmark.runner import manifest, run_benchmark


def environment(cpu_name='Machine A CPU', gpu_name='Machine A GPU'):
    return dict(timestamp_utc='2026-10-22T05:00:00+00:00', platform='Example OS', python='3.11.11 test build',
                packages={'torch': '2.6.0'}, hardware=dict(
                    collected_at_utc='2026-10-22T05:00:00+00:00',
                    cpu=dict(model=cpu_name, physical_cores=8, logical_processors=16, affinity_count=8),
                    ram=dict(total_bytes=16 * 1024**3, scope='runtime-visible RAM, not physical host RAM'),
                    gpu=dict(name=gpu_name, device_index=1, uuid='GPU-example',
                             total_memory_bytes=12 * 1024**3, driver_version='600.00',
                             compute_capability='8.9', cuda_runtime_version='12.4'),
                    windows_host=None, warnings=[]))


class HardwareReportingTests(unittest.TestCase):
    def render(self, metadata):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            stage = dict(stage='01-search-01streams', phase='search', stream_count=1,
                         passed=True, streams=[], resources={})
            write_run_report(path, dict(hardware_result=True, max_stable_streams=None,
                confirmed=False, status='running', stages=[stage], config=Config().to_dict(),
                environment=metadata))
            with (path / 'summary.csv').open(encoding='utf-8-sig') as file:
                rows = list(csv.DictReader(file))
            return (path / 'report.md').read_text(encoding='utf-8'), rows

    def test_each_report_uses_its_saved_machine_not_fixed_3060_notes(self):
        first, _ = self.render(environment())
        second, _ = self.render(environment('Machine B CPU', 'Machine B GPU'))
        self.assertIn('Machine A CPU', first)
        self.assertIn('Machine A GPU', first)
        self.assertIn('Machine B CPU', second)
        self.assertIn('Machine B GPU', second)
        self.assertNotIn('Machine A GPU', second)
        self.assertIn('12.00 GiB', second)
        self.assertIn('runtime-visible', second)
        self.assertIn('未提供', second)
        self.assertIn('3.11.11 test build', second)

    def test_windows_physical_ram_is_rendered_separately(self):
        metadata = environment()
        metadata['hardware']['windows_host'] = dict(source='Windows CIM via run.ps1',
            collected_at_utc='2026-10-22T05:00:00+00:00', cpu={'name': 'Host CPU'},
            ram={'installed_bytes': 32 * 1024**3, 'visible_total_bytes': 31 * 1024**3},
            os={'caption': 'Windows 11'}, warnings=[])
        report, _ = self.render(metadata)
        self.assertIn('32.00 GiB', report)
        self.assertIn('16.00 GiB', report)
        self.assertIn('Windows 11', report)

    def test_comparison_csv_contains_the_same_machine(self):
        _, rows = self.render(environment())
        self.assertEqual(rows[0]['gpu_name'], 'Machine A GPU')
        self.assertEqual(rows[0]['cpu_model'], 'Machine A CPU')
        self.assertEqual(int(rows[0]['runtime_ram_total_bytes']), 16 * 1024**3)

    def test_legacy_report_does_not_look_up_the_viewers_machine(self):
        report, _ = self.render({})
        self.assertIn('未記錄', report)
        self.assertNotIn('i7-11800H', report)

    def test_manifest_collects_runtime_without_initializing_a_gpu(self):
        data = manifest(Config(backend='synthetic'))
        self.assertIn('hardware', data)
        self.assertIsNone(data['hardware']['gpu'])
        self.assertGreater(data['hardware']['ram']['total_bytes'], 0)

    def test_worker_selected_gpu_reaches_saved_environment_and_csv(self):
        selected_gpu = environment(gpu_name='Worker GPU B')['hardware']['gpu']
        with tempfile.TemporaryDirectory() as directory:
            config = Config(backend='synthetic', gpu_device=1, output=directory,
                            duration_s=1, target_fps=5, warmup_s=0)
            def worker(command, **kwargs):
                stage_dir = Path(command[command.index('--stage-dir') + 1])
                write_json(stage_dir / 'result.json', dict(
                    stream_count=1, duration_s=1, passed=True, fatal=False, reasons=[],
                    hardware_result=False, completed_frames=5, total_pipeline_fps=5,
                    completed_model_frame_evaluations=30, model_invocations=30,
                    streams=[dict(stream_id=0, offered=5, completed=5, fps=5,
                                  drop_ratio=0, latency_p95_ms=20, min_window_fps=5)],
                    preflight=[], resources={'samples': 1}, performance_breakdown={},
                    model_environment={'gpu': selected_gpu}))
                return type('Process', (), {'returncode': 0})()
            # Replace only the external worker process, preserving real manifest,
            # result loading/merging, JSON serialization and report/CSV writing.
            with patch('benchmark.runner.subprocess.run', side_effect=worker):
                summary = run_benchmark(config, smoke_streams=1)
            path = Path(summary['run_directory'])
            saved = json.loads((path / 'environment.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['hardware']['gpu'], selected_gpu)
            self.assertEqual(summary['environment']['hardware']['gpu'], selected_gpu)
            with (path / 'summary.csv').open(encoding='utf-8-sig') as file:
                self.assertEqual(next(csv.DictReader(file))['gpu_name'], 'Worker GPU B')


if __name__ == '__main__':
    unittest.main()
