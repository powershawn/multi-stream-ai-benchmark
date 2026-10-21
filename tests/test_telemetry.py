import unittest

from benchmark.telemetry import normalize_gpu_uuid


class TelemetryTests(unittest.TestCase):
    def test_cuda_uuid_is_normalized_for_nvml_without_changing_mig_ids(self):
        raw = '01234567-89ab-cdef-0123-456789abcdef'
        self.assertEqual(normalize_gpu_uuid(raw), 'GPU-' + raw)
        self.assertEqual(normalize_gpu_uuid('GPU-' + raw), 'GPU-' + raw)
        self.assertEqual(normalize_gpu_uuid('MIG-' + raw), 'MIG-' + raw)


if __name__ == '__main__':
    unittest.main()
