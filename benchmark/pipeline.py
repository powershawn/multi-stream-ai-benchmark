"""Bounded real-time readers feeding one fair, batched six-model scheduler."""

from dataclasses import dataclass
import math
import threading
import time

from .metrics import StreamStats


@dataclass
class Frame:
    stream_id: int
    captured_s: float
    image: object


class LatestSlot:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def take(self):
        with self._lock:
            value, self._frame = self._frame, None
            return value


class VideoReader:
    def __init__(self, path, target_fps):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f'cannot open video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_index = 0
        if not math.isfinite(self.fps) or self.fps <= 0 or self.frames <= 0:
            self.close()
            raise ValueError('video must have valid FPS and frame-count metadata')
        if target_fps > self.fps + .01:
            self.close()
            raise ValueError(f'target_fps {target_fps} exceeds source FPS {self.fps:.3f}')

    def read(self, seconds):
        wanted = int(seconds * self.fps) % self.frames
        if wanted < self.next_index:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            self.next_index = 0
        while self.next_index <= wanted:
            if not self.cap.grab():
                raise RuntimeError(f'video decode failed at frame {self.next_index}')
            self.next_index += 1
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            raise RuntimeError(f'video retrieve failed at frame {wanted}')
        return frame

    def close(self):
        self.cap.release()


class SyntheticModels:
    def __init__(self, config):
        self.delay = config.synthetic_delay_ms / 1000
        self.preflight = [dict(model=i, status='synthetic_only') for i in range(6)]

    def infer(self, images):
        times = []
        for _ in range(6):
            start = time.perf_counter()
            time.sleep(self.delay * len(images))
            times.append((time.perf_counter() - start) * 1000)
        return times


def _window(config, count, models, duration, monitor=None):
    slots = [LatestSlot() for _ in range(count)]
    stats = [StreamStats(i, duration, config.target_fps, config.window_s) for i in range(count)]
    readers, threads, errors = [], [], []
    stop = threading.Event()
    try:
        if config.backend == 'yolo':
            for _ in range(count):
                readers.append(VideoReader(config.video, config.target_fps))
        else:
            readers = [None] * count
        # All readers are opened before the common clock starts.
        origin = time.perf_counter() + .1
        deadline = origin + duration
        offered = math.ceil(duration * config.target_fps - 1e-9)

        def produce(index):
            try:
                tick = 0
                while tick < offered and not stop.is_set():
                    captured = tick / config.target_fps
                    if stop.wait(max(0, origin + captured - time.perf_counter())):
                        return
                    # A stalled decoder cannot accumulate an unbounded backlog.
                    latest = min(offered - 1, int(max(0, time.perf_counter() - origin) * config.target_fps))
                    tick = max(tick, latest)
                    captured = tick / config.target_fps
                    image = readers[index].read(captured) if readers[index] else tick
                    if time.perf_counter() >= deadline:
                        return
                    slots[index].put(Frame(index, captured, image))
                    tick += 1
            except Exception as exc:
                errors.append(f'stream {index}: {type(exc).__name__}: {exc}')
                stop.set()

        for i in range(count):
            thread = threading.Thread(target=produce, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)
        if monitor:
            monitor.start(origin, deadline)
        cursor, invocations = 0, 0
        while time.perf_counter() < deadline and not stop.is_set():
            batch = []
            for offset in range(count):
                index = (cursor + offset) % count
                frame = slots[index].take()
                if frame is not None:
                    batch.append(frame)
                if len(batch) == config.batch_size:
                    break
            if not batch:
                stop.wait(.001)
                continue
            cursor = (batch[-1].stream_id + 1) % count
            model_ms = models.infer([frame.image for frame in batch])
            invocations += 6
            completed = time.perf_counter() - origin
            for frame in batch:
                stats[frame.stream_id].record(frame.captured_s, completed, model_ms)
        if errors:
            raise RuntimeError('; '.join(errors))
        return stats, invocations
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        alive = any(thread.is_alive() for thread in threads)
        # Never release a capture while its decode thread is still using it.
        if not alive:
            for reader in readers:
                if reader:
                    reader.close()
        if monitor:
            monitor.stop()
        if alive:
            raise RuntimeError('video decoder did not stop within 10 seconds')


def run_pipeline(config, count, models=None, monitor=None, on_stats=None):
    config.validate()
    if count < 1:
        raise ValueError('stream count must be >= 1')
    if models is None:
        if config.backend == 'synthetic':
            models = SyntheticModels(config)
        else:
            from .yolo import YoloModels
            models = YoloModels(config, max_batch=min(count, config.batch_size))
    if config.warmup_s > 0:
        _window(config, count, models, config.warmup_s)
    if hasattr(models, 'reset_measurements'):
        models.reset_measurements()
    stats, calls = _window(config, count, models, config.duration_s, monitor)
    if on_stats:
        on_stats(stats)
    streams = [item.summary() for item in stats]
    completed = sum(item['completed'] for item in streams)
    model_latency = []
    from .metrics import percentile
    for model_index in range(6):
        timings = [row[2][model_index] for item in stats for row in item.records
                   if row[2][model_index] is not None]
        model_latency.append(dict(model=model_index, batch_latency_p95_ms=percentile(timings, 95)))
    return dict(stream_count=count, duration_s=config.duration_s, streams=streams,
                completed_frames=completed, total_pipeline_fps=completed / config.duration_s,
                completed_model_frame_evaluations=completed * 6, model_invocations=calls,
                per_model=model_latency, preflight=models.preflight,
                model_timing_scope=getattr(models, 'timing_scope', 'predict_wall'),
                performance_breakdown=models.performance_summary() if hasattr(models, 'performance_summary') else {},
                hardware_result=config.backend == 'yolo')
