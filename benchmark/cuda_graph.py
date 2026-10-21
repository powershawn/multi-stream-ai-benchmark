"""Capture six independent forwards; reuse fixed buffers, never cached predictions."""


class CudaGraphExecutor:
    def __init__(self, networks, example, max_batch=1):
        import torch
        if len(networks) != 6 or len({id(model) for model in networks}) != 6:
            raise ValueError('exactly six independent model objects are required')
        parameter_sets = [{p.data_ptr() for p in model.parameters()} for model in networks]
        for index, pointers in enumerate(parameter_sets):
            if any(pointers & previous for previous in parameter_sets[:index]):
                raise ValueError('six independent parameter storage sets are required')
        if not example.is_cuda or example.shape[0] != 1:
            raise ValueError('a single-image CUDA tensor is required for capture')
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError('max_batch must be a positive integer')
        self.torch, self.networks, self.states = torch, networks, {}
        self.capture_count = 0
        with torch.inference_mode():
            for count in range(1, max_batch + 1):
                static = example.repeat(count, *([1] * (example.ndim - 1)))
                stream = torch.cuda.Stream(device=example.device)
                stream.wait_stream(torch.cuda.current_stream(example.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward(static)
                torch.cuda.current_stream(example.device).wait_stream(stream)
                torch.cuda.synchronize(example.device)
                # YOLO reassigns cached anchors when batch shape changes. Retain their
                # tensors so an older graph cannot point at freed/reused allocations.
                constants = [value for model in networks for module in model.modules()
                             for value in vars(module).values()
                             if isinstance(value, torch.Tensor) and value.is_cuda]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = self._forward(static)
                self.states[self._key(static)] = dict(input=static, graph=graph, outputs=outputs,
                    constants=constants, start=torch.cuda.Event(enable_timing=True),
                    end=torch.cuda.Event(enable_timing=True))
                self.capture_count += 1
                graph.replay()
                torch.cuda.synchronize(example.device)

    @staticmethod
    def _key(tensor):
        return tuple(tensor.shape), tensor.dtype, tensor.device

    def _forward(self, tensor):
        outputs = []
        for model in self.networks:
            value = model(tensor)
            outputs.append(value[0] if isinstance(value, (tuple, list)) else value)
        return outputs

    def replay(self, tensor):
        state = self.states.get(self._key(tensor))
        if state is None:
            raise ValueError(f'input shape/dtype/device was not prepared before measurement: {tuple(tensor.shape)}')
        with self.torch.inference_mode():
            state['input'].copy_(tensor, non_blocking=True)
            state['start'].record()
            state['graph'].replay()
            state['end'].record()
        return state['outputs'], state['start'], state['end']
