"""FPS counts frames that complete ALL six models, never individual detections."""

from dataclasses import dataclass, field
import math
import statistics


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


@dataclass
class StreamStats:
    stream_id: int
    duration_s: float
    target_fps: float
    window_s: float
    records: list = field(default_factory=list)

    def record(self, captured_s, completed_s, model_ms):
        if 0 <= captured_s <= completed_s < self.duration_s:
            self.records.append((captured_s, completed_s, model_ms))

    def summary(self):
        offered = math.ceil(self.duration_s * self.target_fps - 1e-9)
        completed = len(self.records)
        latencies = [(end - start) * 1000 for start, end, _ in self.records]
        # Include empty windows; retain the last partial window with its true duration.
        windows = []
        for index in range(math.ceil(self.duration_s / self.window_s)):
            start = index * self.window_s
            end = min(start + self.window_s, self.duration_s)
            frames = sum(start <= completion < end for _, completion, _ in self.records)
            windows.append(dict(start_s=start, end_s=end, fps=frames / (end - start)))
        return dict(stream_id=self.stream_id, offered=offered, completed=completed,
                    fps=completed / self.duration_s, drop_ratio=(offered - completed) / offered,
                    latency_mean_ms=statistics.mean(latencies) if latencies else None,
                    latency_p50_ms=percentile(latencies, 50), latency_p95_ms=percentile(latencies, 95),
                    latency_p99_ms=percentile(latencies, 99),
                    min_window_fps=min(w['fps'] for w in windows), windows=windows)


def summarize_telemetry(samples):
    def values(key):
        return [row[key] for row in samples if row.get(key) is not None]
    def peak(key):
        return max(values(key), default=None)
    def mean(key):
        found = values(key)
        return statistics.mean(found) if found else None
    ratios = [row['gpu_memory_used_mb'] / row['gpu_memory_total_mb'] * 100
              for row in samples if row.get('gpu_memory_used_mb') is not None and row.get('gpu_memory_total_mb')]
    return dict(samples=len(samples), gpu_util_mean_percent=mean('gpu_util_percent'),
                gpu_util_peak_percent=peak('gpu_util_percent'),
                gpu_memory_peak_mb=peak('gpu_memory_used_mb'),
                gpu_memory_peak_percent=max(ratios, default=None),
                cpu_mean_percent=mean('cpu_percent'), cpu_peak_percent=peak('cpu_percent'),
                ram_peak_mb=peak('ram_used_mb'), ram_peak_percent=peak('ram_percent'),
                process_rss_peak_mb=peak('process_rss_mb'))
