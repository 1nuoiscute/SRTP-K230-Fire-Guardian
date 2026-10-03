"""Queue admission boundaries and overlap regression checks, without network or inference."""
import unittest
from fetch_kitchen_context_queue import license_allowed,known_titles
from pathlib import Path
import json,tempfile

class QueueBoundaryTests(unittest.TestCase):
    def test_restrictive_license_is_not_accidentally_admitted(self):
        for name in ['CC BY-NC 4.0','CC BY-ND 4.0','CC BY-SA 4.0 or other terms','Public domain']:
            self.assertFalse(license_allowed(name,'https://creativecommons.org/licenses/by/4.0/'))
    def test_license_name_url_must_agree(self):
        self.assertTrue(license_allowed('CC BY-SA 4.0','https://creativecommons.org/licenses/by-sa/4.0/'))
        self.assertFalse(license_allowed('CC BY-SA 4.0','https://creativecommons.org/licenses/by/4.0'))
        self.assertFalse(license_allowed('CC BY 2.0','https://creativecommons.org.evil.test/licenses/by/2.0/'))
        self.assertTrue(license_allowed('CC0','https://creativecommons.org/publicdomain/zero/1.0/'))
    def test_nested_lineage_and_url_alias_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'
            p.write_text(json.dumps({'lineage':[{'source_page':'https://commons.wikimedia.org/wiki/File:Foo_bar.jpg'},{'source_page':'https://commons.wikimedia.org/wiki/File:Foo%20bar.jpg'}]}),encoding='utf-8')
            self.assertEqual(known_titles([p]),['File:Foo bar.jpg'])
    def test_changed_candidate_fails_before_output(self):
        try:
            from audit_kitchen_context_queue import audit
        except ImportError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'image.jpg').write_bytes(b'changed')
            (d/'manifest.json').write_text(json.dumps([{'file':'image.jpg','sha256':'wrong'}]),encoding='utf-8')
            (d/'plan.json').write_text(json.dumps({'roots':[],'csv_manifests':[]}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Candidate bytes changed'):
                audit(d,d/'plan.json',d/'out')
            self.assertFalse((d/'out').exists())


class FrozenRoleTests(unittest.TestCase):
    def row(self,**changes):
        r=dict(file='one.jpg',role='development',visually_reviewed=True,source_group='group_a',note='reviewed',author='author_a',original_sha1='source_a',visible_target='visible_no_flame',boxes=[],width=100,height=80)
        r.update(changes);return r
    def test_related_source_group_cannot_cross_train_and_holdout(self):
        from freeze_kitchen_context_review import validate_rows
        with self.assertRaisesRegex(ValueError,'crosses frozen roles'):
            validate_rows([self.row(),self.row(file='two.jpg',role='reserved_holdout',author='author_b',original_sha1='source_b')])
    def test_shared_author_and_alias_cannot_cross_roles(self):
        from freeze_kitchen_context_review import validate_rows
        for changes in [dict(author='author_a',original_sha1='source_b'),dict(author='author_b',original_sha1='source_a')]:
            with self.assertRaisesRegex(ValueError,'crosses frozen roles'):
                validate_rows([self.row(),self.row(file='two.jpg',role='reserved_holdout',source_group='group_b',**changes)])
    def test_unknown_target_must_not_become_empty_negative(self):
        from freeze_kitchen_context_review import validate_rows
        with self.assertRaisesRegex(ValueError,'Unknown visible target'):
            validate_rows([self.row(visible_target='unknown')])
        validate_rows([self.row(visible_target='unknown',role='held')])
    def test_box_must_be_finite_and_inside_actual_image(self):
        from freeze_kitchen_context_review import validate_rows
        for box in [[0,0,101,30],[0,0,float('nan'),30]]:
            with self.assertRaisesRegex(ValueError,'Invalid box'):
                validate_rows([self.row(visible_target='flame',boxes=[box])])
    def test_positive_must_have_visible_flame_box(self):
        from freeze_kitchen_context_review import validate_rows
        with self.assertRaisesRegex(ValueError,'requires reviewed boxes'):
            validate_rows([self.row(visible_target='flame')])

if __name__=='__main__': unittest.main()
