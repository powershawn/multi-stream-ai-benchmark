"""Sample host/VM CPU and RAM, process RSS, and whole-device GPU resources."""

import csv
import os
import threading
import time

from .metrics import summarize_telemetry


def normalize_gpu_uuid(value):
    value = str(value)
    return value if value.startswith(('GPU-', 'MIG-')) else 'GPU-' + value


class TelemetryMonitor:
    def __init__(self, config, output):
        import psutil
        self.psutil, self.config, self.output = psutil, config, output
        self.process = psutil.Process()
        self.samples, self.warnings = [], []
        self.done = threading.Event()
        self.thread = None
        self.nvml = self.handle = None
        if config.backend == 'yolo':
            try:
                import pynvml
                import torch
                pynvml.nvmlInit()
                self.nvml = pynvml
                props = torch.cuda.get_device_properties(config.gpu_device)
                uuid = getattr(props, 'uuid', None)
                if uuid:
                    self.handle = pynvml.nvmlDeviceGetHandleByUUID(normalize_gpu_uuid(uuid))
                else:
                    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
                    identity = visible[config.gpu_device].strip() if len(visible) > config.gpu_device and visible[0] else str(config.gpu_device)
                    if identity.startswith(('GPU-', 'MIG-')):
                        self.handle = pynvml.nvmlDeviceGetHandleByUUID(identity)
                    else:
                        self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(identity))
            except Exception as exc:
                self.warn(f'NVML unavailable: {exc}; GPU memory falls back to CUDA, utilization may be null')

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def sample(self, elapsed):
        ram = self.psutil.virtual_memory()
        row = dict(elapsed_s=elapsed, cpu_percent=self.psutil.cpu_percent(),
                   ram_percent=ram.percent, ram_used_mb=ram.used / 1024**2,
                   process_rss_mb=self.process.memory_info().rss / 1024**2,
                   gpu_util_percent=None, gpu_memory_used_mb=None, gpu_memory_total_mb=None,
                   gpu_temperature_c=None, gpu_power_w=None)
        if self.handle is not None:
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                row.update(gpu_memory_used_mb=memory.used / 1024**2, gpu_memory_total_mb=memory.total / 1024**2)
            except Exception as exc:
                self.warn(f'NVML memory unavailable: {exc}')
            try:
                row['gpu_util_percent'] = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            except Exception as exc:
                self.warn(f'GPU utilization unavailable: {exc}')
            try:
                row['gpu_temperature_c'] = self.nvml.nvmlDeviceGetTemperature(self.handle, 0)
                row['gpu_power_w'] = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            except Exception:
                pass  # Optional sensors are unsupported on some devices.
        if row['gpu_memory_used_mb'] is None and self.config.backend == 'yolo':
            try:
                import torch
                free, total = torch.cuda.mem_get_info(self.config.gpu_device)
                row.update(gpu_memory_used_mb=(total - free) / 1024**2, gpu_memory_total_mb=total / 1024**2)
            except Exception as exc:
                self.warn(f'CUDA memory telemetry unavailable: {exc}')
        return row

    def start(self, origin, deadline):
        def collect():
            self.psutil.cpu_percent()  # Prime non-blocking CPU counter.
            if self.done.wait(max(0, origin - time.perf_counter())):
                return
            with self.output.open('w', newline='', encoding='utf-8') as file:
                writer = None
                while not self.done.is_set() and time.perf_counter() < deadline:
                    try:
                        row = self.sample(time.perf_counter() - origin)
                        self.samples.append(row)
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        file.flush()
                    except Exception as exc:
                        self.warn(f'telemetry sample failed: {exc}')
                    self.done.wait(min(self.config.sample_interval_s, max(0, deadline - time.perf_counter())))
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass

    def summary(self):
        result = summarize_telemetry(self.samples)
        result['warnings'] = self.warnings
        result['scope'] = 'whole GPU; CPU/RAM from Linux host or Docker VM; process RSS separate'
        return result
