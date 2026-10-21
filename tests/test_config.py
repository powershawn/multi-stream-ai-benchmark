import json
from pathlib import Path
import tempfile
import unittest

from benchmark.config import Config


class ConfigDocumentationTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            return Config.load(path)

    def test_help_is_documentation_not_a_runtime_setting(self):
        config = self.load({'target_fps': 10, '_help': {'target_fps': '每路分析 FPS'}})
        self.assertEqual(config.target_fps, 10)
        self.assertNotIn('_help', config.to_dict())

    def test_unknown_settings_are_still_rejected(self):
        with self.assertRaises(TypeError):
            self.load({'target_fpps': 10, '_help': {}})

    def test_every_project_setting_is_documented(self):
        data = json.loads(Path('config.json').read_text(encoding='utf-8'))
        self.assertIn('_help', data)
        self.assertEqual(set(data['_help']), set(Config().to_dict()))
        self.assertEqual(set(data) - {'_help'}, set(Config().to_dict()))
        self.assertTrue(all(isinstance(value, str) and value.strip()
                            for value in data['_help'].values()))
        self.assertEqual(self.load(data).target_fps, 10)

    def test_legacy_configuration_keeps_automatic_search(self):
        self.assertIsNone(self.load({}).test_streams)

    def test_accepts_ascending_counts_and_a_single_count(self):
        for counts in [[1, 2, 4, 8], [2], [32, 64]]:
            with self.subTest(counts=counts):
                config = Config(test_streams=counts).validate()
                self.assertEqual(config.to_dict()['test_streams'], counts)

    def test_rejects_ambiguous_or_invalid_counts(self):
        for counts in [[], '1,2', [0], [-1], [True], [1.5], [2, 2], [4, 2], [1, None]]:
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'test_streams'):
                Config(test_streams=counts).validate()


if __name__ == '__main__':
    unittest.main()
