"""Earlier reserved roles must survive new queue admission, before any copies."""
import json
from pathlib import Path
import tempfile
import unittest
from data_integrity import sha256
from freeze_inherited_kitchen_review import validate_inheritance


class InheritedKitchenRolesTests(unittest.TestCase):
    def row(self, **changes):
        r=dict(file='old.jpg',role='reserved_holdout',visually_reviewed=True,
               source_group='kitchen_a',note='reviewed',author='author_a',
               original_sha1='original_a',visible_target='visible_no_flame',
               boxes=[],width=100,height=80)
        r.update(changes)
        return r

    def fixture(self, folder, child):
        parent=folder/'parent.json'
        parent.write_text(json.dumps({'rows':[self.row()]}),encoding='utf-8')
        plan=folder/'plan.json'
        plan.write_text(json.dumps({'inherited_reviews':[{'path':str(parent),'sha256':sha256(parent)}]}),encoding='utf-8')
        return parent,plan,{'inheritance_plan_sha256':sha256(plan),'rows':[child]}

    def test_new_queue_cannot_move_old_group_to_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan,review=self.fixture(Path(tmp),self.row(file='new.jpg',role='development',author='new_author',original_sha1='new_original'))
            with self.assertRaisesRegex(ValueError,'crosses frozen roles'):
                validate_inheritance(review,plan)

    def test_author_or_original_alias_cannot_bypass_inherited_group(self):
        for changes in [dict(author=' AUTHOR_A ',original_sha1='new_original'),dict(author='new_author',original_sha1='original_a')]:
            with tempfile.TemporaryDirectory() as tmp:
                _,plan,review=self.fixture(Path(tmp),self.row(file='new.jpg',role='development',source_group='new_group',**changes))
                with self.assertRaisesRegex(ValueError,'crosses frozen roles'):
                    validate_inheritance(review,plan)

    def test_changed_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,plan,review=self.fixture(Path(tmp),self.row(file='new.jpg'))
            parent.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Inherited review changed'):
                validate_inheritance(review,plan)

    def test_changed_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan,review=self.fixture(Path(tmp),self.row(file='new.jpg'))
            plan.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Inheritance plan changed'):
                validate_inheritance(review,plan)

    def test_empty_parent_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'plan.json'
            plan.write_text(json.dumps({'inherited_reviews':[]}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Missing inherited reviews'):
                validate_inheritance({'inheritance_plan_sha256':sha256(plan),'rows':[self.row()]},plan)

    def test_related_queue_keeps_reserved_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,plan,review=self.fixture(Path(tmp),self.row(file='new.jpg'))
            self.assertEqual(len(validate_inheritance(review,plan)),1)


if __name__ == '__main__': unittest.main()
