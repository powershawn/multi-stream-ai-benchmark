"""Configuration is saved with each run so results can be compared."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Config:
    video: str = 'video.mp4'
    weights: list = field(default_factory=lambda: ['models/yolo11m.pt'] * 6)
    output: str = 'results'
    gpu_device: int = 0
    target_fps: float = 15.0
    imgsz: int = 640
    half: bool = True
    batch_size: int = 1
    torch_threads: int = 1
    duration_s: float = 60.0
    warmup_s: float = 5.0
    max_streams: int = 16
    test_streams: object = None
    confirm_duration_s: float = 300.0
    confirm_repeats: int = 3
    window_s: float = 5.0
    min_fps_ratio: float = .95
    min_window_fps_ratio: float = .90
    max_latency_p95_ms: float = 500.0
    max_drop_ratio: float = .01
    max_vram_percent: float = 90.0
    max_ram_percent: float = 90.0
    max_cpu_percent: object = None
    sample_interval_s: float = 1.0
    startup_timeout_s: float = 180.0
    backend: str = 'yolo'
    execution: str = 'cuda_graph'
    synthetic_delay_ms: float = 2.0

    def validate(self):
        positive = ['target_fps', 'imgsz', 'batch_size', 'torch_threads', 'duration_s',
                    'max_streams', 'confirm_duration_s', 'confirm_repeats', 'window_s',
                    'max_latency_p95_ms', 'sample_interval_s', 'startup_timeout_s']
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        for name in ['gpu_device', 'imgsz', 'batch_size', 'torch_threads', 'max_streams', 'confirm_repeats']:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be an integer >= 0')
        if self.test_streams is not None:
            if (not isinstance(self.test_streams, list) or not self.test_streams
                    or any(type(count) is not int or count < 1 for count in self.test_streams)
                    or any(left >= right for left, right in zip(self.test_streams, self.test_streams[1:]))):
                raise ValueError('test_streams must be null or a nonempty, strictly increasing list of positive integers')
        for name in ['warmup_s', 'synthetic_delay_ms']:
            if not isinstance(getattr(self, name), (int, float)) or not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and >= 0')
        for name in ['min_fps_ratio', 'min_window_fps_ratio']:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if not 0 <= self.max_drop_ratio < 1:
            raise ValueError('max_drop_ratio must be in [0, 1)')
        for name in ['max_vram_percent', 'max_ram_percent', 'max_cpu_percent']:
            value = getattr(self, name)
            if value is not None and not 0 < value <= 100:
                raise ValueError(f'{name} must be in (0, 100]')
        if not isinstance(self.weights, list) or len(self.weights) != 6 or any(not isinstance(w, str) or not w for w in self.weights):
            raise ValueError('weights must contain exactly six nonempty model paths')
        if self.backend not in ('yolo', 'synthetic'):
            raise ValueError('backend must be yolo or synthetic')
        if self.execution not in ('predict', 'cuda_graph'):
            raise ValueError('execution must be predict or cuda_graph')
        if type(self.half) is not bool:
            raise ValueError('half must be true or false')
        if self.duration_s * self.target_fps < 1:
            raise ValueError('duration_s * target_fps must be >= 1')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        # Reserve one documentation key; keep unknown runtime settings strict.
        data.pop('_help', None)
        return cls(**data).validate()
