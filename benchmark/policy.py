"""Per-camera acceptance and bounded capacity search, independent of hardware."""


def assess(streams, metrics, config):
    reasons = []
    if not streams:
        reasons.append('no stream results')
    for stream in streams:
        prefix = f"stream {stream['stream_id']}"
        if stream['fps'] < config.target_fps * config.min_fps_ratio:
            reasons.append(f'{prefix}: FPS below target tolerance')
        if stream['min_window_fps'] < config.target_fps * config.min_window_fps_ratio:
            reasons.append(f'{prefix}: window FPS below stability threshold')
        if stream['latency_p95_ms'] is None or stream['latency_p95_ms'] > config.max_latency_p95_ms:
            reasons.append(f'{prefix}: p95 latency missing or above limit')
        if stream['drop_ratio'] > config.max_drop_ratio:
            reasons.append(f'{prefix}: dropped/incomplete frame ratio above limit')
    for key, limit in [('gpu_memory_peak_percent', config.max_vram_percent),
                       ('ram_peak_percent', config.max_ram_percent),
                       ('cpu_peak_percent', config.max_cpu_percent)]:
        if limit is None:
            continue
        value = metrics.get(key)
        if value is None:
            reasons.append(f'{key}: telemetry unavailable; cannot validate')
        elif value > limit:
            reasons.append(f'{key}: {value:.1f}% exceeds {limit:.1f}%')
    if not metrics.get('samples'):
        reasons.append('no telemetry samples')
    return not reasons, reasons


def search_capacity(run, maximum):
    best, failed = 0, None
    count = 1
    while True:
        if run(count)['passed']:
            best = count
            if count == maximum:
                return dict(best=best, at_limit=True)
            count = min(count * 2, maximum)
        else:
            failed = count
            break
    low, high = best + 1, failed - 1
    while low <= high:
        count = (low + high) // 2
        if run(count)['passed']:
            best, low = count, count + 1
        else:
            high = count - 1
    return dict(best=best, at_limit=False)


def search_configured_counts(run, counts):
    """Visit only the ascending configured counts, stopping at the first failure."""
    best = 0
    for count in counts:
        if not run(count)['passed']:
            return dict(best=best, at_limit=False)
        best = count
    return dict(best=best, at_limit=True)


def confirm_capacity(run, candidate, repeats, reduce_on_failure=True):
    while candidate > 0:
        for _ in range(repeats):
            if not run(candidate)['passed']:
                if not reduce_on_failure:
                    return 0
                candidate -= 1
                break
        else:
            return candidate
    return 0
