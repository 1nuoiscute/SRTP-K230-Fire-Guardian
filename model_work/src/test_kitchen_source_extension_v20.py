"""Source sampling and reserved identities at the training admission boundary."""
import unittest
from build_kitchen_source_extension_v20 import development_rows, reject_forbidden, reject_repeated_source_images


class SourceExtensionTests(unittest.TestCase):
    def row(self,**changes):
        r=dict(file='new.jpg',role='development',visually_reviewed=True,source_group='new_group',note='reviewed',
               author='new_author',original_sha1='new_original',visible_target='visible_no_flame',boxes=[],
               width=100,height=80,sha256='new_digest')
        r.update(changes);return r

    def test_repeated_author_group_shares_negative_budget(self):
        rows=[self.row(),self.row(file='two.jpg',sha256='other_digest',original_sha1='other_original')]
        self.assertEqual([r['weight'] for r in development_rows(rows,8,4)],[2,2])

    def test_positive_budget_and_reserved_exclusion(self):
        rows=[self.row(visible_target='flame',boxes=[[1,2,10,20]]),self.row(file='reserved.jpg',role='reserved_holdout',source_group='reserved_group',author='reserved_author',original_sha1='reserved_original',sha256='reserved_digest')]
        chosen=development_rows(rows,8,4)
        self.assertEqual([(r['file'],r['weight']) for r in chosen],[('new.jpg',8)])

    def test_nondivisible_group_budget_is_rejected(self):
        rows=[self.row(file=f'{i}.jpg',original_sha1=str(i),sha256=str(i)) for i in range(3)]
        with self.assertRaisesRegex(ValueError,'cannot be split evenly'):development_rows(rows,8,4)

    def test_mixed_visibility_same_group_cannot_silently_change_sampling(self):
        with self.assertRaisesRegex(ValueError,'Mixed or unknown'):
            development_rows([self.row(),self.row(file='positive.jpg',visible_target='flame',boxes=[[1,2,10,20]])],8,4)

    def test_reserved_bytes_group_author_original_cannot_enter_training(self):
        reserved=self.row(role='reserved_holdout')
        for item in [{'sha256':'new_digest'}, {'sha256':'other','source_group':'new_group'}, {'sha256':'other','author':' NEW_AUTHOR '}, {'sha256':'other','original_sha1':'new_original'}]:
            with self.assertRaisesRegex(ValueError,'Reserved or held identity'):
                reject_forbidden([reserved],[item])

    def test_parent_validation_snapshot_schema_and_duplicate_new_bytes(self):
        parent={'lineage':[{'sha256':'train'}],'validation':[{'sha256':'validation','label_sha256':'label'}]}
        for rows in [[{'sha256':'train'}],[{'sha256':'validation'}],[{'sha256':'new'},{'sha256':'new'}]]:
            with self.assertRaisesRegex(ValueError,'Repeated or parent/validation'):
                reject_repeated_source_images(rows,parent)
        reject_repeated_source_images([{'sha256':'new'}],parent)

    def test_completed_training_rejects_wrong_dataset(self):
        from train_kitchen_source_head_v20 import validate_completion
        with self.assertRaisesRegex(ValueError,'Wrong completed v20 data'):
            validate_completion({'dataset':{'manifest_sha256':'different'}},[],{})

    def test_v20_schedule_matches_v19_except_data_and_run_name(self):
        from pathlib import Path
        from train_kitchen_source_head_v20 import frozen_schedule
        from train_context_flame_head_v19 import frozen_schedule as old_schedule
        try:
            previous=old_schedule(Path('v19/data.yaml'),Path('runs/v19'),Path('v16/best.pt'),23)
            current=frozen_schedule(Path('v20/data.yaml'),Path('runs/v20'),Path('v16/best.pt'),23)
        except ImportError as error:self.skipTest(str(error))
        self.assertEqual({k for k in current if current[k]!=previous[k]},{'data','name'})


if __name__=='__main__':unittest.main()
