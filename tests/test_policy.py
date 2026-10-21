import unittest

from benchmark.config import Config
from benchmark import policy
from benchmark.policy import assess, search_capacity, confirm_capacity


def good_stream(index=0, **changes):
    value = dict(stream_id=index, offered=150, completed=150, fps=15.0,
                 drop_ratio=0.0, latency_p95_ms=100.0, min_window_fps=15.0)
    value.update(changes)
    return value


def good_metrics():
    return dict(gpu_memory_peak_percent=50.0, ram_peak_percent=40.0,
                cpu_peak_percent=60.0, samples=2)


class PolicyTests(unittest.TestCase):
    def test_configured_counts_stop_at_first_failure_without_refinement(self):
        self.assertTrue(hasattr(policy, 'search_configured_counts'))
        calls = []
        def run(count):
            calls.append(count)
            return {'passed': count < 4}
        result = policy.search_configured_counts(run, [1, 2, 4, 8])
        self.assertEqual(calls, [1, 2, 4])
        self.assertEqual(result['best'], 2)
        self.assertFalse(result['at_limit'])

    def test_first_configured_failure_and_all_pass_are_reported(self):
        self.assertTrue(hasattr(policy, 'search_configured_counts'))
        calls = []
        def fail(count):
            calls.append(count)
            return {'passed': False}
        self.assertEqual(policy.search_configured_counts(fail, [2, 4, 8])['best'], 0)
        self.assertEqual(calls, [2])
        result = policy.search_configured_counts(lambda n: {'passed': True}, [2, 4])
        self.assertEqual(result, dict(best=4, at_limit=True))

    def test_confirmation_can_stop_without_trying_unlisted_counts(self):
        calls = []
        def fail(count):
            calls.append(count)
            return {'passed': False}
        self.assertEqual(confirm_capacity(fail, 4, 3, reduce_on_failure=False), 0)
        self.assertEqual(calls, [4])

    def test_healthy_camera_passes_at_the_users_fifteen_fps(self):
        self.assertEqual(Config().target_fps, 15)
        self.assertEqual(assess([good_stream()], good_metrics(), Config()), (True, []))

    def test_each_camera_must_pass_even_when_total_fps_is_high(self):
        passed, reasons = assess([good_stream(), good_stream(1, fps=2)], good_metrics(), Config())
        self.assertFalse(passed)
        self.assertTrue(any('stream 1' in reason and 'FPS' in reason for reason in reasons))

    def test_latency_drops_windows_and_resource_limits_are_enforced(self):
        cases = [dict(latency_p95_ms=501), dict(drop_ratio=.02), dict(min_window_fps=8)]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertFalse(assess([good_stream(**changes)], good_metrics(), Config())[0])
        metrics = good_metrics()
        metrics['gpu_memory_peak_percent'] = 91
        self.assertFalse(assess([good_stream()], metrics, Config())[0])

    def test_missing_required_telemetry_never_passes(self):
        metrics = good_metrics()
        metrics['gpu_memory_peak_percent'] = None
        self.assertFalse(assess([good_stream()], metrics, Config())[0])
        self.assertFalse(assess([], good_metrics(), Config())[0])

    def test_step_up_then_refine_to_exact_integer_capacity(self):
        calls = []
        def run(count):
            calls.append(count)
            return {'passed': count <= 11}
        result = search_capacity(run, 16)
        self.assertEqual(result['best'], 11)
        self.assertEqual(calls[:5], [1, 2, 4, 8, 16])
        self.assertEqual(set(calls[5:]), {12, 10, 11})
        self.assertFalse(result['at_limit'])

    def test_zero_pass_and_upper_bound_are_reported_truthfully(self):
        self.assertEqual(search_capacity(lambda n: {'passed': False}, 16)['best'], 0)
        result = search_capacity(lambda n: {'passed': True}, 16)
        self.assertEqual(result['best'], 16)
        self.assertTrue(result['at_limit'])

    def test_confirmation_reduces_count_after_any_failed_repeat(self):
        calls = []
        def run(n):
            calls.append(n)
            return {'passed': n < 3 or calls.count(3) == 1}
        self.assertEqual(confirm_capacity(run, 3, 3), 2)
        self.assertEqual(calls, [3, 3, 2, 2, 2])

    def test_reject_invalid_configuration(self):
        for changes in [dict(target_fps=0), dict(weights=['one.pt']), dict(duration_s=-1),
                        dict(max_streams=0), dict(gpu_device=-1), dict(batch_size=0),
                        dict(max_drop_ratio=2), dict(confirm_repeats=0)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes).validate()


if __name__ == '__main__':
    unittest.main()
