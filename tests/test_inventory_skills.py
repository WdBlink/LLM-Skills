import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inventory_skills import inventory


class InventoryTests(unittest.TestCase):
    def test_unknown_without_metadata_and_symlink_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source').mkdir()
            (root/'source/SKILL.md').write_text('---\nname: source\n---\n')
            (root/'alias').symlink_to(root/'source', target_is_directory=True)
            result = inventory([root])
            self.assertEqual(len(result['skills']), 1)
            self.assertEqual(result['skills'][0]['origin'], 'unknown')
            self.assertEqual(len(result['skills'][0]['locations']), 2)
            locked = inventory([root], {'skills': {'source': {'source': 'other/repo'}}})
            self.assertEqual(locked['skills'][0]['origin'], 'third_party_install')

    def test_package_overrides_owner_lock_and_stale_review_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'installed'; package = Path(tmp)/'package'
            (root/'tool').mkdir(parents=True); package.mkdir()
            (root/'tool/SKILL.md').write_text('changed')
            (package/'tool').symlink_to(root/'tool', target_is_directory=True)
            result = inventory([root], {'skills': {'tool': {'source': 'WdBlink/fork'}}},
                               {'upstream': package}, catalog={'skills': {'tool': {
                                   'skill_md_sha256': 'old', 'classification': 'locally_maintained', 'evidence': ['stale']}}})
            item = result['skills'][0]
            self.assertEqual(item['origin'], 'third_party_package')
            self.assertFalse(item['review_matches_entry'])
            self.assertNotIn('stale', item['evidence'])

    def test_later_alias_records_are_merged_and_conflicts_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'installed'; source = Path(tmp)/'real-skill'
            root.mkdir(); source.mkdir(); (source/'SKILL.md').write_text('entry')
            for alias in ['a-alias', 'z-alias']:
                (root/alias).symlink_to(source, target_is_directory=True)
            lock = {'skills': {'z-alias': {'source': 'other/repo'}}}
            result = inventory([root], lock)
            self.assertEqual(result['skills'][0]['installer_source'], 'other/repo')
            self.assertEqual(result['skills'][0]['origin'], 'third_party_install')
            lock['skills']['a-alias'] = {'source': 'WdBlink/fork'}
            result = inventory([root], lock)
            self.assertEqual(result['skills'][0]['origin'], 'conflicting_install_records')
            self.assertEqual(result['skills'][0]['installer_sources'], ['WdBlink/fork', 'other/repo'])
            self.assertIsNone(result['skills'][0]['installer_source'])

    def test_owner_repo_is_not_original_authorship(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['fork', 'download']:
                (root/name).mkdir(); (root/name/'SKILL.md').write_text('same')
            result = inventory([root], {'skills': {
                'fork': {'source': 'WdBlink/fork'}, 'download': {'source': 'other/repo'}}})
            self.assertEqual(result['counts'], {'third_party_install': 1, 'owner_repository': 1})
            self.assertEqual(len(result['skills']), 2)


if __name__ == '__main__':
    unittest.main()
