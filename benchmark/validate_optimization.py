"""Compare all six real model outputs on changed frames and changing batches."""

import json
from pathlib import Path


def main():
    import torch
    from .config import Config
    from .pipeline import VideoReader
    from .yolo import YoloModels

    config = Config.load('config.json')
    config.execution = 'cuda_graph'
    models = YoloModels(config, max_batch=2)
    reader = VideoReader(config.video, config.target_fps)
    try:
        frames = [reader.read(seconds) for seconds in (0, 3, 30, 60, 88)]
    finally:
        reader.close()
    checks = []
    # Return to batch 1 after batch 2, and to an earlier frame after later frames.
    for indices in [(0,), (1, 2), (3,), (4, 0), (0,)]:
        images = [frames[index] for index in indices]
        expected = []
        for model in models.models:
            results = models._predict(model, images)
            expected.append([result.boxes.data.clone() for result in results])
        models.infer(images)
        row = dict(frame_indices=list(indices), model_checks=[])
        for index, (actual, reference) in enumerate(zip(models.last_detections, expected)):
            if len(actual) != len(images):
                raise AssertionError('missing a frame in an independent model output')
            errors = []
            for actual_boxes, reference_boxes in zip(actual, reference):
                torch.testing.assert_close(actual_boxes, reference_boxes, rtol=0, atol=0)
                errors.append((actual_boxes - reference_boxes).abs().max().item() if actual_boxes.numel() else 0)
            row['model_checks'].append(dict(model=index, detections=[len(boxes) for boxes in actual],
                                            max_abs_errors=errors))
        checks.append(row)
        print(json.dumps(row), flush=True)
    path = Path('results/diagnosis/output_equivalence.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(passed=True, precision='same FP16, bit-exact detection tensors',
        independent_models=6, graph_captures=models.graph.capture_count, checks=checks), indent=2), encoding='utf-8')
    print(f'PASS: all six models matched exactly on {sum(len(row["frame_indices"]) for row in checks)} image evaluations each', flush=True)


if __name__ == '__main__':
    main()
