"""Fresh per-run hardware snapshots; never initialize CUDA in the parent process."""

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform


HOST_INFO_ENV = 'BENCHMARK_HOST_INFO_B64'


def cpu_model_from_proc(text):
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition(':')
        if separator:
            fields.setdefault(key.strip().lower(), value.strip())
    for key in ('model name', 'hardware', 'cpu model'):
        if fields.get(key):
            return fields[key]
    return None


def read_host_snapshot(payload, now=None):
    """Reject old copied environment values, rather than identifying the wrong host."""
    if not payload:
        return None, []
    try:
        if len(payload) > 65536:
            raise ValueError('host snapshot exceeds size limit')
        data = json.loads(base64.b64decode(payload, validate=True).decode('utf-8'))
        if (not isinstance(data, dict) or data.get('schema_version') != 1
                or data.get('source') != 'Windows CIM via run.ps1'):
            raise ValueError('unrecognized host snapshot schema/source')
        collected = datetime.fromisoformat(data['collected_at_utc'].replace('Z', '+00:00'))
        if collected.tzinfo is None:
            raise ValueError('host snapshot must contain a timezone')
        age = ((now or datetime.now(timezone.utc)) - collected).total_seconds()
        if not -60 <= age <= 600:
            raise ValueError('host snapshot is stale or from the future')
        if any(not isinstance(data.get(key), dict) for key in ('cpu', 'ram', 'os')):
            raise ValueError('incomplete host snapshot')
        # Whitelist fields: do not copy arbitrary environment data into reports.
        result = {key: data.get(key) for key in ('schema_version', 'collected_at_utc', 'source')}
        for key, fields in [('cpu', ('name', 'physical_cores', 'logical_processors')),
                            ('ram', ('installed_bytes', 'visible_total_bytes')),
                            ('os', ('caption', 'version'))]:
            result[key] = {field: data[key].get(field) for field in fields}
        for key, fields in [('cpu', ('physical_cores', 'logical_processors')),
                            ('ram', ('installed_bytes', 'visible_total_bytes'))]:
            for field in fields:
                number = result[key][field]
                if number is not None:
                    if (type(number) not in (int, float) or not 0 < number < 2**63
                            or number != int(number)):
                        raise ValueError(f'invalid numeric host field: {key}.{field}')
                    result[key][field] = int(number)
        for key, field in [('cpu', 'name'), ('os', 'caption'), ('os', 'version')]:
            if result[key][field] is not None and not isinstance(result[key][field], str):
                raise ValueError(f'invalid text host field: {key}.{field}')
        result['warnings'] = [str(value) for value in data.get('warnings', [])]
        return result, []
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
        return None, [f'Windows host snapshot ignored: {exc}']


def collect_runtime_hardware(environ=None):
    import psutil
    environ = os.environ if environ is None else environ
    warnings = []
    model = None
    try:
        if platform.system() == 'Linux':
            model = cpu_model_from_proc(Path('/proc/cpuinfo').read_text(encoding='utf-8'))
        elif platform.system() == 'Windows':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                model = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except (OSError, ValueError) as exc:
        warnings.append(f'CPU model unavailable: {exc}')
    model = model or platform.processor() or None
    affinity = None
    try:
        affinity = len(psutil.Process().cpu_affinity())
    except (AttributeError, NotImplementedError, psutil.Error):
        pass
    host, host_warnings = read_host_snapshot(environ.get(HOST_INFO_ENV))
    warnings.extend(host_warnings)
    ram_total = None
    try:
        ram_total = psutil.virtual_memory().total
    except (OSError, psutil.Error) as exc:
        warnings.append(f'Runtime RAM capacity unavailable: {exc}')
    return dict(schema_version=1, collected_at_utc=datetime.now(timezone.utc).isoformat(),
                cpu=dict(model=model, physical_cores=psutil.cpu_count(logical=False),
                         logical_processors=psutil.cpu_count(logical=True), affinity_count=affinity),
                ram=dict(total_bytes=ram_total,
                         scope='runtime-visible RAM (OS/Docker VM); not necessarily physical host RAM'),
                gpu=None, windows_host=host, warnings=warnings)


def describe_cuda_device(props, device_index, cuda_version, driver_version):
    from .telemetry import normalize_gpu_uuid
    uuid = getattr(props, 'uuid', None)
    return dict(device_index=device_index, name=props.name,
                uuid=normalize_gpu_uuid(uuid) if uuid else None,
                total_memory_bytes=props.total_memory,
                compute_capability=f'{props.major}.{props.minor}',
                cuda_runtime_version=cuda_version, driver_version=driver_version,
                source='actual selected CUDA device in the inference worker')


def collect_cuda_hardware(torch, device_index):
    """Called only by the existing GPU worker, before measured inference."""
    if not torch.cuda.is_available():
        return None
    if not 0 <= device_index < torch.cuda.device_count():
        return None
    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(device_index)
    driver = None
    nvml = None
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode('utf-8')
    except Exception:
        pass  # Driver version is optional; never invent it from another run.
    finally:
        if nvml:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    return describe_cuda_device(props, device_index, torch.version.cuda, driver)
