import unittest

from benchmark.config import Config
from benchmark.cuda_graph import CudaGraphExecutor

try:
    import torch
    CUDA_GRAPHS = torch.cuda.is_available() and hasattr(torch.cuda, 'CUDAGraph')
except ImportError:
    CUDA_GRAPHS = False


class GraphContractTests(unittest.TestCase):
    def test_invalid_execution_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'execution'):
            Config(execution='skip_five_models').validate()


@unittest.skipUnless(CUDA_GRAPHS, 'requires CUDA and CUDAGraph support')
class CudaGraphTests(unittest.TestCase):
    def models(self):
        class Affine(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(float(scale), device='cuda'))
            def forward(self, value):
                return value * self.scale + self.scale
        return [Affine(i).eval() for i in range(1, 7)]

    def test_every_replay_updates_six_independent_model_outputs(self):
        networks = self.models()
        example = torch.ones(1, 3, 8, 8, device='cuda')
        executor = CudaGraphExecutor(networks, example, max_batch=2)
        for count, value in [(1, 3), (2, 7), (1, 11)]:
            data = torch.full((count, 3, 8, 8), value, device='cuda', dtype=torch.float32)
            outputs, start, end = executor.replay(data)
            torch.cuda.synchronize()
            self.assertEqual(len(outputs), 6)
            with torch.inference_mode():
                for actual, model in zip(outputs, networks):
                    torch.testing.assert_close(actual, model(data))
            self.assertGreaterEqual(start.elapsed_time(end), 0)
        self.assertEqual(executor.capture_count, 2)

    def test_measurement_cannot_silently_capture_an_unprepared_shape(self):
        example = torch.ones(1, 3, 8, 8, device='cuda')
        executor = CudaGraphExecutor(self.models(), example, max_batch=1)
        with self.assertRaisesRegex(ValueError, 'prepared'):
            executor.replay(torch.ones(2, 3, 8, 8, device='cuda'))

    def test_six_calls_cannot_reuse_one_model_object(self):
        models = self.models()
        example = torch.ones(1, 3, 8, 8, device='cuda')
        with self.assertRaisesRegex(ValueError, 'independent'):
            CudaGraphExecutor([models[0]] * 6, example)


if __name__ == '__main__':
    unittest.main()
