"""Isolated stage processes recover GPU memory after OOM, crashes or timeouts."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

from .config import Config
from .policy import assess, confirm_capacity, search_capacity, search_configured_counts
from .reporting import write_json, write_run_report, write_stage_details


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(config):
    metadata = dict(python=sys.version, platform=platform.platform(),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    cpu_count=os.cpu_count(), packages={}, files={})
    for package in ['torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'psutil', 'nvidia-ml-py']:
        try:
            metadata['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if config.backend == 'yolo':
        for name in [config.video] + list(dict.fromkeys(config.weights)):
            path = Path(name)
            if not path.is_file():
                raise FileNotFoundError(f'missing {name}; check video path or run prepare')
            metadata['files'][name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
        try:
            query = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'],
                                   capture_output=True, text=True, timeout=10)
            metadata['nvidia_smi'] = (query.stdout or query.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            metadata['nvidia_smi'] = str(exc)
    return metadata


def execute_stage(config, count, directory):
    from .pipeline import SyntheticModels, run_pipeline
    from .telemetry import TelemetryMonitor
    started = time.perf_counter()
    monitor = None
    try:
        if config.backend == 'yolo':
            from .yolo import YoloModels
            models = YoloModels(config, max_batch=min(count, config.batch_size))
        else:
            models = SyntheticModels(config)
        write_json(directory / 'preflight.json', models.preflight)
        monitor = TelemetryMonitor(config, directory / 'resources.csv')
        stats = []
        result = run_pipeline(config, count, models, monitor, stats.extend)
        result['resources'] = monitor.summary()
        result['model_environment'] = getattr(models, 'metadata', {})
        # A synthetic run validates software only; it can exercise search logic without a GPU.
        acceptance = replace(config, max_vram_percent=None) if config.backend == 'synthetic' else config
        result['passed'], result['reasons'] = assess(result['streams'], result['resources'], acceptance)
        result['fatal'] = False
        write_stage_details(directory, result, stats)
    except Exception as exc:
        traceback.print_exc()
        message = f'{type(exc).__name__}: {exc}'
        capacity_error = any(text in message.lower() for text in ['out of memory', 'cublas_status_alloc_failed'])
        result = dict(stream_count=count, passed=False, reasons=[message],
                      fatal=not capacity_error, hardware_result=config.backend == 'yolo')
        if monitor:
            monitor.stop()
            result['resources'] = monitor.summary()
    result['wall_time_s'] = time.perf_counter() - started
    write_json(directory / 'result.json', result)
    return result


def run_benchmark(config, smoke_streams=None):
    config.validate()
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    directory = Path(config.output).resolve() / run_id
    directory.mkdir(parents=True)
    config_path = directory / 'config.json'
    write_json(config_path, config.to_dict())
    metadata = manifest(config)
    write_json(directory / 'environment.json', metadata)
    stages = []
    summary = dict(hardware_result=config.backend == 'yolo', max_stable_streams=None,
                   at_limit=False, confirmed=False, status='running', stages=stages,
                   config=config.to_dict(), environment=metadata, run_directory=str(directory))
    write_run_report(directory, summary)

    def run_case(count, phase='search'):
        sequence = len(stages) + 1
        label = f'{sequence:02d}-{phase}-{count:02d}streams'
        stage_dir = directory / label
        stage_dir.mkdir()
        stage_config = replace(config, duration_s=config.confirm_duration_s) if phase == 'confirm' else config
        stage_config_path = stage_dir / 'config.json'
        write_json(stage_config_path, stage_config.to_dict())
        print(f'[{label}] {count} cameras x 6 models @ {config.target_fps} FPS; '
              f'{stage_config.duration_s}s measured + {config.warmup_s}s warmup', flush=True)
        command = [sys.executable, '-m', 'benchmark', '_stage', '--config', str(stage_config_path),
                   '--streams', str(count), '--stage-dir', str(stage_dir)]
        timeout = stage_config.startup_timeout_s + stage_config.warmup_s + stage_config.duration_s + 30
        with (stage_dir / 'stage.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8'))
                result_path = stage_dir / 'result.json'
                if not result_path.exists():
                    result = dict(stream_count=count, passed=False, fatal=True,
                                  reasons=[f'worker exited {process.returncode} without a result; see stage.log'])
                else:
                    import json
                    result = json.loads(result_path.read_text(encoding='utf-8'))
            except subprocess.TimeoutExpired:
                result = dict(stream_count=count, passed=False,
                              fatal=not (stage_dir / 'preflight.json').exists(),
                              reasons=[f'worker timeout after {timeout:.1f}s'])
                write_json(stage_dir / 'result.json', result)
        result.update(stage=label, phase=phase)
        stages.append(result)
        write_run_report(directory, summary)
        print(f"[{label}] {'PASS' if result['passed'] else 'FAIL'}: "
              f"{result.get('total_pipeline_fps', 0):.2f} full-pipeline FPS; "
              + '; '.join(result.get('reasons', [])), flush=True)
        if result.get('fatal'):
            raise RuntimeError(f'infrastructure or software failure at {label}; see {stage_dir / "stage.log"}')
        return result

    try:
        if smoke_streams is not None:
            result = run_case(smoke_streams, 'smoke')
            summary.update(status='smoke_passed' if result['passed'] else 'smoke_failed',
                           smoke_passed=result['passed'])
        else:
            configured = config.test_streams is not None
            search = (search_configured_counts(run_case, config.test_streams) if configured
                      else search_capacity(run_case, config.max_streams))
            summary['search_candidate_streams'] = search['best']
            best = confirm_capacity(lambda n: run_case(n, 'confirm'), search['best'], config.confirm_repeats,
                                    reduce_on_failure=not configured)
            hardware = config.backend == 'yolo'
            summary.update(max_stable_streams=best if hardware else None, confirmed=hardware,
                           at_limit=bool(search['at_limit'] and best == search['best'] and best > 0),
                           status=('confirmed' if best else 'no_passing_streams') if hardware else 'synthetic_only')
    except BaseException:
        summary['status'] = 'aborted'
        raise
    finally:
        write_run_report(directory, summary)
        print(f'Report: {directory / "report.md"}', flush=True)
    return summary
