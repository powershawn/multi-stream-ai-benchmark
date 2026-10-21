"""Machine-readable CSV/JSON and a readable report, without extra dependencies."""

import csv
import json


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_stage_details(directory, result, stats):
    write_json(directory / 'preflight.json', result['preflight'])
    with (directory / 'frames.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['stream_id', 'capture_s', 'complete_s', 'end_to_end_ms'] + [f'model_{i}_batch_ms' for i in range(6)])
        for stream in stats:
            for start, end, timings in stream.records:
                writer.writerow([stream.stream_id, start, end, (end - start) * 1000] + timings)


def write_run_report(directory, summary):
    write_json(directory / 'summary.json', summary)
    columns = ['stage', 'phase', 'streams', 'passed', 'total_fps', 'worst_camera_fps',
               'worst_p95_ms', 'worst_drop_ratio', 'gpu_util_mean_percent', 'gpu_memory_peak_mb',
               'gpu_memory_peak_percent', 'cpu_mean_percent', 'ram_peak_mb', 'ram_peak_percent', 'reasons']
    rows = []
    for stage in summary['stages']:
        streams, metrics = stage.get('streams', []), stage.get('resources', {})
        latencies = [s['latency_p95_ms'] for s in streams if s['latency_p95_ms'] is not None]
        row = dict(stage=stage['stage'], phase=stage['phase'], streams=stage['stream_count'],
                   passed=stage['passed'], total_fps=stage.get('total_pipeline_fps'),
                   worst_camera_fps=min((s['fps'] for s in streams), default=None),
                   worst_p95_ms=max(latencies, default=None),
                   worst_drop_ratio=max((s['drop_ratio'] for s in streams), default=None),
                   reasons='; '.join(stage.get('reasons', [])))
        row.update({key: metrics.get(key) for key in columns if key in metrics})
        rows.append(row)
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if not summary['hardware_result']:
        conclusion = 'SYNTHETIC SOFTWARE TEST ONLY. No GPU camera capacity was measured.'
    elif summary.get('status') == 'aborted':
        conclusion = 'Run aborted. No maximum stable camera capacity was confirmed; inspect the stage logs.'
    elif summary.get('status') == 'running':
        conclusion = 'Run in progress. Camera capacity has not yet been confirmed.'
    elif not summary.get('confirmed'):
        conclusion = 'Smoke test only. No maximum stable camera capacity is claimed.'
    elif summary['max_stable_streams'] == 0:
        conclusion = ('No stable camera count completed the required confirmation; inspect individual stages.'
                      if summary.get('search_candidate_streams', 0) > 0 else
                      'No tested camera count passed the configured acceptance criteria.')
    elif summary['config'].get('test_streams') is not None:
        conclusion = (f"Highest confirmed count among configured test_streams: {summary['max_stable_streams']} cameras; "
                      'this is not an absolute hardware maximum. Only configured counts were considered; '
                      'the list stops at the first failure and intermediate counts are not refined.')
    elif summary['at_limit']:
        conclusion = f"Confirmed at least {summary['max_stable_streams']} cameras; this is not an absolute hardware maximum. Increase max_streams to continue."
    else:
        conclusion = f"Maximum confirmed stable count in this search: {summary['max_stable_streams']} cameras."
    config = summary['config']
    plan = (f"Configured test_streams: {config['test_streams']}. Stop the list at its first failure; "
            'then confirm the highest passing count. A failed confirmation ends the run without fallback.'
            if config.get('test_streams') is not None else
            f"Automatic search up to {config['max_streams']} streams, with integer refinement after a failed step.")
    lines = ['# Six-model camera benchmark', '', conclusion, '',
             plan, '',
             f"Each camera: {config['target_fps']} analysis FPS; six independent models; input size {config['imgsz']}; FP16={config['half']}; batch={config['batch_size']}.", '',
             f"Acceptance: per-camera FPS >= {config['target_fps'] * config['min_fps_ratio']:.2f}; p95 <= {config['max_latency_p95_ms']} ms; drop ratio <= {config['max_drop_ratio']:.2%}; VRAM <= {config['max_vram_percent']}%; RAM <= {config['max_ram_percent']}%.", '',
             'FPS is a frame completing all six models. Latency includes scheduled capture, decode, queue, preprocess, inference and postprocess. Uncompleted frames count as drops.', '',
             'The search assumes capacity degrades as stream count increases. It is specific to this video, software, batching, sampling rate and acceptance settings. Resource peaks are sampled, not continuous hardware maxima.', '',
             '| Stage | Phase | Cameras | Pass | Total FPS | Worst camera FPS | Worst p95 ms |',
             '|---|---|---:|---|---:|---:|---:|']
    def fmt(value):
        return '-' if value is None else f'{value:.2f}'
    for row in rows:
        lines.append(f"| {row['stage']} | {row['phase']} | {row['streams']} | {row['passed']} | {fmt(row['total_fps'])} | {fmt(row['worst_camera_fps'])} | {fmt(row['worst_p95_ms'])} |")
    lines.extend(['', 'See summary.json for configuration, file hashes, environment, individual-camera results and failure reasons. See each stage for frames.csv, resources.csv, preflight.json and stage.log.'])
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
