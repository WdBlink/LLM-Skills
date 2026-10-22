import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('mindmap', Path(__file__).resolve().parents[1]/'scripts/update_mindmap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VaultTests(unittest.TestCase):
    def test_missing_configuration_does_not_choose_another_users_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, 'home', return_value=Path(tmp)), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Set --vault'):
                module.resolve_vault(None)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_explicit_then_environment_then_config(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, 'home', return_value=Path(tmp)), patch.dict(os.environ, {'MIND_MAP_VAULT': tmp+'/env'}, clear=True):
            self.assertEqual(module.resolve_vault(tmp+'/explicit'), (Path(tmp)/'explicit').resolve())
            self.assertEqual(module.resolve_vault(None), (Path(tmp)/'env').resolve())
            os.environ.pop('MIND_MAP_VAULT')
            config = Path(tmp)/'.agent-mindmap/config.json'
            config.parent.mkdir()
            config.write_text(json.dumps({'defaultVaultPath': tmp+'/config'}))
            self.assertEqual(module.resolve_vault(None), (Path(tmp)/'config').resolve())
            config.write_text('{broken')
            with self.assertRaisesRegex(ValueError, 'Cannot read'):
                module.resolve_vault(None)


if __name__ == '__main__':
    unittest.main()
