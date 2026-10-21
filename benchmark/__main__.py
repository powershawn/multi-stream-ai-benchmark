"""CLI: prepare weights, verify one camera, then search and confirm capacity."""

import argparse
import os
from pathlib import Path
import shutil
import sys

from .config import Config


def prepare(config):
    from ultralytics import YOLO
    for weight in dict.fromkeys(config.weights):
        target = Path(weight).resolve()
        if target.is_file():
            print(f'Already present: {target}', flush=True)
            continue
        if target.name != 'yolo11m.pt':
            raise FileNotFoundError(f'custom weights must be supplied by user: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download to the mounted model directory, not the container filesystem.
        original = Path.cwd()
        try:
            os.chdir(target.parent)
            model = YOLO(target.name)
            downloaded = Path(model.ckpt_path).resolve()
            if downloaded != target:
                shutil.copy2(downloaded, target)
        finally:
            os.chdir(original)
        print(f'Ready: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Six YOLO-M models per camera GPU capacity benchmark')
    parser.add_argument('command', choices=['prepare', 'smoke', 'run', '_stage'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--video')
    parser.add_argument('--output')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--target-fps', type=float)
    parser.add_argument('--max-streams', type=int)
    parser.add_argument('--confirm-duration', type=float)
    parser.add_argument('--confirm-repeats', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--backend', choices=['yolo', 'synthetic'])
    parser.add_argument('--execution', choices=['predict', 'cuda_graph'])
    parser.add_argument('--streams', type=int, default=1)
    parser.add_argument('--stage-dir', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = Config.load(args.config)
        for option, field in [('video', 'video'), ('output', 'output'), ('duration', 'duration_s'),
                              ('warmup', 'warmup_s'), ('target_fps', 'target_fps'),
                              ('execution', 'execution'), ('max_streams', 'max_streams'), ('confirm_duration', 'confirm_duration_s'),
                              ('confirm_repeats', 'confirm_repeats'), ('batch_size', 'batch_size'), ('backend', 'backend')]:
            value = getattr(args, option)
            if value is not None:
                setattr(config, field, value)
        config.validate()
        if args.streams < 1:
            raise ValueError('--streams must be >= 1')
        if args.command == 'prepare':
            prepare(config)
            return 0
        from .runner import execute_stage, run_benchmark
        if args.command == '_stage':
            if not args.stage_dir:
                raise ValueError('--stage-dir is required for an internal worker')
            result = execute_stage(config, args.streams, Path(args.stage_dir))
            return 2 if result.get('fatal') else 0
        summary = run_benchmark(config, args.streams if args.command == 'smoke' else None)
        if args.command == 'smoke':
            return 0 if summary['smoke_passed'] else 1
        return 1 if summary['max_stable_streams'] == 0 else 0
    except KeyboardInterrupt:
        print('Interrupted. Partial results were retained.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
