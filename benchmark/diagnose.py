"""Evidence-producing microbenchmarks; never reports camera capacity."""

import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from .config import Config
    from .pipeline import VideoReader
    from .yolo import YoloModels

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', default='results/diagnosis')
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    config = Config.load('config.json')
    config.execution = 'predict'
    models = YoloModels(config)
    reader = VideoReader(config.video, config.target_fps)
    decode_ms, frames = [], []
    try:
        for index in range(30):
            start = time.perf_counter()
            frames.append(reader.read(index / config.target_fps))
            decode_ms.append((time.perf_counter() - start) * 1000)
    finally:
        reader.close()
    image = frames[0]
    images = [image] * args.batch_size
    predictor = models.models[0].predictor
    networks = [model.model for model in models.models]
    rows = {}

    def measure(name, action):
        for _ in range(3):
            action()
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            action()
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        rows[name] = dict(mean_ms=statistics.mean(timings), median_ms=statistics.median(timings),
                          min_ms=min(timings), max_ms=max(timings))
        print(name, rows[name], flush=True)

    with torch.inference_mode():
        measure('original_six_predict', lambda: models.infer(images))
        rows['original_sdk_components_ms'] = {
            key: sum(model.predictor.results[0].speed[key] for model in models.models)
            for key in ('preprocess', 'inference', 'postprocess')}
        tensor = predictor.preprocess(images)

        def six_forward():
            return [network(tensor) for network in networks]

        def nms(outputs):
            return [non_max_suppression(output[0].clone(), conf_thres=.25, iou_thres=.7, max_det=300)
                    for output in outputs]

        measure('preprocess_once', lambda: predictor.preprocess(images))
        measure('eager_six_forward_only', six_forward)
        eager_outputs = six_forward()
        measure('six_nms_only', lambda: nms(eager_outputs))

        def shared_eager():
            shared = predictor.preprocess(images)
            return [non_max_suppression(network(shared)[0], conf_thres=.25, iou_thres=.7, max_det=300)
                    for network in networks]

        measure('shared_preprocess_eager_full', shared_eager)
        # Capture the same six INDEPENDENT networks, with one static input buffer.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                six_forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = six_forward()
        graph.replay()
        torch.cuda.synchronize()
        differences = [(actual[0] - expected[0]).abs().max().item()
                       for actual, expected in zip(graph_outputs, eager_outputs)]
        rows['graph_raw_max_abs_difference_each_model'] = differences
        measure('graph_six_forward_only', graph.replay)

        def graph_full():
            tensor.copy_(predictor.preprocess(images))
            graph.replay()
            return nms(graph_outputs)

        measure('graph_shared_preprocess_full', graph_full)
        # An inference on a different image must update all six graph outputs.
        tensor.copy_(predictor.preprocess([frames[15]] * args.batch_size))
        graph.replay()
        torch.cuda.synchronize()
        eager_changed = six_forward()
        torch.cuda.synchronize()
        rows['graph_changed_frame_max_abs_difference_each_model'] = [
            (actual[0] - expected[0]).abs().max().item()
            for actual, expected in zip(graph_outputs, eager_changed)]
        rows['input_shape'] = list(tensor.shape)
        rows['video_decode'] = dict(mean_ms=statistics.mean(decode_ms), max_ms=max(decode_ms))
        # Profiler evidence is separate from uninstrumented wall-clock measurements.
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    models.infer(images)
            (directory / 'original_profiler.txt').write_text(
                prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25), encoding='utf-8')
            prof.export_chrome_trace(str(directory / 'original_trace.json'))
            events = prof.events()
            rows['original_profile_per_frame'] = dict(
                cuda_kernel_launch_calls=sum('cudaLaunchKernel' in event.name for event in events) / 2,
                sync_calls=sum('synchronize' in event.name.lower() for event in events) / 2)
        except Exception as exc:
            rows['profiler_error'] = str(exc)
    (directory / 'microbench.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
