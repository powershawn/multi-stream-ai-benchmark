"""Render saved hardware only; viewing an old report must not inspect this host."""


def comparison_fields(metadata):
    hardware = metadata.get('hardware') or {}
    gpu = hardware.get('gpu') or {}
    cpu = hardware.get('cpu') or {}
    ram = hardware.get('ram') or {}
    host = hardware.get('windows_host') or {}
    return dict(gpu_name=gpu.get('name'), gpu_uuid=gpu.get('uuid'),
                gpu_driver_version=gpu.get('driver_version'), cpu_model=cpu.get('model'),
                cpu_logical_processors=cpu.get('logical_processors'),
                runtime_ram_total_bytes=ram.get('total_bytes'),
                windows_host_ram_installed_bytes=(host.get('ram') or {}).get('installed_bytes'))


def hardware_lines(metadata):
    def value(text):
        return ('未記錄' if text is None else str(text)).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')
    def gib(number):
        return '未記錄' if number is None else f'{number / 1024**3:.2f} GiB'
    hardware = metadata.get('hardware') or {}
    cpu, ram, gpu = (hardware.get(key) or {} for key in ('cpu', 'ram', 'gpu'))
    rows = [
        ('測試開始時間（UTC）', metadata.get('timestamp_utc')),
        ('設備資訊擷取時間（UTC）', hardware.get('collected_at_utc')),
        ('測試環境 OS', metadata.get('platform')),
        ('Python', metadata.get('python')),
        ('CPU 型號（測試環境可見）', cpu.get('model')),
        ('CPU 實體核心／邏輯 CPU（測試環境可見）',
         f"{value(cpu.get('physical_cores'))} / {value(cpu.get('logical_processors'))}"),
        ('程序 CPU affinity 可用數', cpu.get('affinity_count')),
        ('RAM（測試環境可見容量）', gib(ram.get('total_bytes'))),
        ('RAM 範圍', ram.get('scope')),
        ('GPU（實際推論裝置）', gpu.get('name')),
        ('CUDA 可見裝置索引／GPU UUID', f"{value(gpu.get('device_index'))} / {value(gpu.get('uuid'))}"),
        ('GPU VRAM 總容量', gib(gpu.get('total_memory_bytes'))),
        ('NVIDIA 驅動／CUDA runtime', f"{value(gpu.get('driver_version'))} / {value(gpu.get('cuda_runtime_version'))}"),
        ('GPU Compute Capability', gpu.get('compute_capability')),
    ]
    for package in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python'):
        rows.append((package, (metadata.get('packages') or {}).get(package)))
    host = hardware.get('windows_host')
    if host:
        rows.extend([
            ('Windows 主機資訊來源／時間', f"{value(host.get('source'))} / {value(host.get('collected_at_utc'))}"),
            ('Windows 主機 CPU', (host.get('cpu') or {}).get('name')),
            ('Windows 主機實體安裝 RAM', gib((host.get('ram') or {}).get('installed_bytes'))),
            ('Windows 主機 OS 可見 RAM', gib((host.get('ram') or {}).get('visible_total_bytes'))),
            ('Windows 主機 OS', (host.get('os') or {}).get('caption')),
        ])
    else:
        rows.append(('Windows 主機實體規格', '未提供；直接 Docker 執行無法據此推算實體主機 RAM'))
    lines = ['## 本次測試設備（自動擷取）', '', '| 項目 | 紀錄 |', '|---|---|']
    lines.extend(f'| {key} | {value(item)} |' for key, item in rows)
    lines.extend(['', '以上只使用此份測試保存的資料，不會沿用其他設備的規格。GPU 資訊在推論 worker 回報後補入；未取得時保留未記錄。',
                  'CPU／RAM 採樣與驗收使用測試 OS／Docker VM 範圍；Windows 主機規格另列，不拿來替換 VM 的分母。', ''])
    for warning in list(hardware.get('warnings') or []) + list((host or {}).get('warnings') or []):
        lines.extend([f'設備資訊警告：{value(warning)}', ''])
    return lines
