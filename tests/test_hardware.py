import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest

from benchmark import hardware


class HardwareTests(unittest.TestCase):
    def test_linux_cpu_model_is_not_a_generic_architecture_string(self):
        text = 'processor : 0\nmodel name : Example CPU 9000\n\nprocessor : 1\nmodel name : Example CPU 9000\n'
        self.assertEqual(hardware.cpu_model_from_proc(text), 'Example CPU 9000')
        self.assertEqual(hardware.cpu_model_from_proc('Hardware : ARM Board\n'), 'ARM Board')
        self.assertIsNone(hardware.cpu_model_from_proc('processor : 0\n'))

    def test_gpu_description_uses_the_selected_cuda_properties(self):
        props = SimpleNamespace(name='Different GPU', total_memory=12 * 1024**3,
                                uuid='1234-abcd', major=8, minor=9)
        gpu = hardware.describe_cuda_device(props, 1, '12.4', '600.00')
        self.assertEqual(gpu['device_index'], 1)
        self.assertEqual(gpu['name'], 'Different GPU')
        self.assertEqual(gpu['uuid'], 'GPU-1234-abcd')
        self.assertEqual(gpu['total_memory_bytes'], 12 * 1024**3)
        self.assertEqual(gpu['compute_capability'], '8.9')

    def test_missing_gpu_uuid_is_not_replaced_with_another_device(self):
        props = SimpleNamespace(name='GPU without UUID', total_memory=1, major=8, minor=6)
        self.assertIsNone(hardware.describe_cuda_device(props, 0, None, None)['uuid'])

    def test_collector_queries_the_requested_nonfirst_device(self):
        selected = []
        props = [SimpleNamespace(name=name, total_memory=1024, uuid=name, major=8, minor=6)
                 for name in ['GPU A', 'GPU B']]
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               set_device=selected.append, get_device_properties=lambda index: props[index])
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.4'))
        result = hardware.collect_cuda_hardware(torch, 1)
        self.assertEqual(selected, [1])
        self.assertEqual(result['name'], 'GPU B')
        self.assertEqual(result['device_index'], 1)

    def host_payload(self, when):
        data = dict(schema_version=1, collected_at_utc=when.isoformat(),
                    source='Windows CIM via run.ps1', cpu=dict(name='Host CPU'),
                    ram=dict(installed_bytes=32 * 1024**3, visible_total_bytes=31 * 1024**3),
                    os=dict(caption='Windows'), warnings=[])
        return base64.b64encode(json.dumps(data).encode()).decode()

    def test_host_snapshot_is_fresh_for_this_invocation(self):
        now = datetime.now(timezone.utc)
        data, warnings = hardware.read_host_snapshot(self.host_payload(now), now=now)
        self.assertEqual(data['cpu']['name'], 'Host CPU')
        self.assertEqual(warnings, [])

    def test_stale_future_or_invalid_snapshot_is_never_reused(self):
        now = datetime.now(timezone.utc)
        for payload in [self.host_payload(now - timedelta(hours=1)),
                        self.host_payload(now + timedelta(hours=1)), 'invalid', 'e30=']:
            with self.subTest(payload=payload[:12]):
                data, warnings = hardware.read_host_snapshot(payload, now=now)
                self.assertIsNone(data)
                self.assertTrue(warnings)
        self.assertEqual(hardware.read_host_snapshot(None), (None, []))

    def test_invalid_host_field_types_cannot_break_report_rendering(self):
        now = datetime.now(timezone.utc)
        data = json.loads(base64.b64decode(self.host_payload(now)))
        data['ram']['installed_bytes'] = '32 GB'
        payload = base64.b64encode(json.dumps(data).encode()).decode()
        result, warnings = hardware.read_host_snapshot(payload, now=now)
        self.assertIsNone(result)
        self.assertTrue(warnings)

    def test_deeply_nested_invalid_snapshot_does_not_abort_a_run(self):
        payload = base64.b64encode(('[' * 1100 + '0' + ']' * 1100).encode()).decode()
        result, warnings = hardware.read_host_snapshot(payload)
        self.assertIsNone(result)
        self.assertTrue(warnings)

    def test_new_runtime_collection_has_explicit_scope_and_no_stale_host(self):
        result = hardware.collect_runtime_hardware(environ={})
        self.assertGreater(result['cpu']['logical_processors'], 0)
        self.assertGreater(result['ram']['total_bytes'], 0)
        self.assertIn('runtime', result['ram']['scope'])
        self.assertIsNone(result['windows_host'])
        self.assertIsNone(result['gpu'])
        self.assertIn('collected_at_utc', result)

    def test_windows_host_collector_exists(self):
        self.assertTrue(Path('scripts/Get-HostInfo.ps1').is_file())

    @unittest.skipUnless(os.name == 'nt' and shutil.which('pwsh'), 'Windows CIM collector')
    def test_real_windows_collector_is_valid_fresh_json(self):
        self.assertTrue(Path('scripts/Get-HostInfo.ps1').is_file())
        process = subprocess.run(['pwsh', '-NoProfile', '-File', 'scripts/Get-HostInfo.ps1', '-AsBase64'],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        data, warnings = hardware.read_host_snapshot(process.stdout.strip())
        self.assertIsNotNone(data, warnings)
        self.assertGreater(data['ram']['visible_total_bytes'], 0)
        self.assertTrue(data['cpu']['name'])


if __name__ == '__main__':
    unittest.main()
