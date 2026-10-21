import unittest
import tempfile
from pathlib import Path

from benchmark.metrics import StreamStats, percentile, summarize_telemetry
from benchmark.pipeline import LatestSlot, Frame, VideoReader, run_pipeline
from benchmark.config import Config


class MetricsTests(unittest.TestCase):
    def test_full_pipeline_fps_is_not_multiplied_by_six(self):
        stats = StreamStats(0, 2, 10, 1)
        for i in range(20):
            stats.record(i / 10, i / 10 + .05, [5] * 6)
        result = stats.summary()
        self.assertEqual(result['fps'], 10)
        self.assertEqual(result['completed'], 20)
        self.assertEqual(result['offered'], 20)
        self.assertEqual(result['drop_ratio'], 0)
        self.assertAlmostEqual(result['latency_p95_ms'], 50)

    def test_incomplete_and_overwritten_frames_count_as_drops(self):
        stats = StreamStats(0, 2, 10, 1)
        stats.record(0, .1, [10] * 6)
        self.assertEqual(stats.summary()['drop_ratio'], .95)
        slot = LatestSlot()
        slot.put(Frame(0, 0, 'old'))
        slot.put(Frame(0, .1, 'new'))
        self.assertEqual(slot.take().image, 'new')
        self.assertIsNone(slot.take())

    def test_deadline_boundary_is_consistent_with_window_accounting(self):
        stats = StreamStats(0, 1, 10, 1)
        stats.record(.9, 1.0, [1] * 6)
        self.assertEqual(stats.summary()['completed'], 0)

    def test_empty_latency_is_unknown_and_empty_windows_are_not_hidden(self):
        stats = StreamStats(0, 2, 10, 1)
        self.assertIsNone(stats.summary()['latency_p95_ms'])
        for i in range(10):
            stats.record(i / 10, i / 10 + .01, [1] * 6)
        self.assertEqual(stats.summary()['min_window_fps'], 0)
        self.assertIsNone(percentile([], 95))

    def test_missing_gpu_utilization_stays_null(self):
        result = summarize_telemetry([dict(cpu_percent=20, ram_percent=30,
            ram_used_mb=1000, process_rss_mb=200, gpu_util_percent=None,
            gpu_memory_used_mb=100, gpu_memory_total_mb=1000)])
        self.assertIsNone(result['gpu_util_mean_percent'])
        self.assertEqual(result['gpu_memory_peak_percent'], 10)

    def test_group_graph_timing_is_not_invented_for_individual_models(self):
        class GroupModels:
            preflight = [{'status': 'ok'}] * 6
            timing_scope = 'group_cuda_graph'
            def infer(self, images):
                return [None] * 6
        config = Config(duration_s=.2, warmup_s=0, target_fps=10, window_s=.2, backend='synthetic')
        result = run_pipeline(config, 1, GroupModels())
        self.assertTrue(all(row['batch_latency_p95_ms'] is None for row in result['per_model']))
        self.assertEqual(result['model_timing_scope'], 'group_cuda_graph')

    def test_real_scheduling_with_synthetic_backend(self):
        config = Config(duration_s=.5, warmup_s=0, target_fps=10, window_s=.25,
                        backend='synthetic', synthetic_delay_ms=0)
        result = run_pipeline(config, 2)
        self.assertEqual(len(result['streams']), 2)
        self.assertEqual([s['offered'] for s in result['streams']], [5, 5])
        self.assertTrue(all(s['completed'] >= 4 for s in result['streams']))
        self.assertEqual(result['completed_model_frame_evaluations'], result['completed_frames'] * 6)
        self.assertFalse(result['hardware_result'])

    def test_inference_finishing_after_deadline_is_not_counted(self):
        config = Config(duration_s=.2, warmup_s=0, target_fps=10, window_s=.2,
                        backend='synthetic', synthetic_delay_ms=60)
        result = run_pipeline(config, 1)
        self.assertEqual(result['completed_frames'], 0)
        self.assertEqual(result['streams'][0]['drop_ratio'], 1)

    def test_video_reader_preserves_timeline_and_loops_at_eof(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'fixture.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for shade in (0, 80, 160):
                writer.write(np.full((32, 32, 3), shade, dtype=np.uint8))
            writer.release()
            reader = VideoReader(path, 5)
            try:
                self.assertLess(reader.read(0).mean(), 5)
                self.assertGreater(reader.read(.2).mean(), 150)
                self.assertLess(reader.read(.3).mean(), 5)
            finally:
                reader.close()
            with self.assertRaisesRegex(ValueError, 'exceeds source FPS'):
                VideoReader(path, 15)


if __name__ == '__main__':
    unittest.main()
