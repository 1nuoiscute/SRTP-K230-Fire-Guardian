"""A missing, partial or changed scratch comparison cannot authorize the next GPU run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch', 'ultralytics', 'cv2', 'numpy', 'yaml'))
if AVAILABLE:
    import train_visible_flame_pilot_v15 as pilot
    from data_integrity import sha256


@unittest.skipUnless(AVAILABLE, 'Optional local ML dependencies unavailable')
class PrerequisiteTests(unittest.TestCase):
    def dump(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def fixture(self, folder):
        weights = folder / 'scratch/weights'
        weights.mkdir(parents=True)
        paths = dict(v3=folder / 'v3.pt', v13_best=folder / 'v13.pt',
                     scratch_v14_best=weights / 'best.pt', scratch_v14_last=weights / 'last.pt')
        for name, path in paths.items():
            path.write_bytes(name.encode())  # identity-only test fixtures, not model checkpoints
        digests = {name: sha256(path) for name, path in paths.items()}
        updates = weights.parent / 'full_network_update_audit.json'
        self.dump(updates, [{'epoch': i + 1} for i in range(100)])
        meta = dict(config={'epochs': 100}, primary_checkpoint_screen=pilot.PRIMARY,
                    completed_audits={updates.name: sha256(updates)})
        meta_path = weights.parent / 'experiment_meta.json'
        self.dump(meta_path, meta)
        plan = dict(data_manifest_sha256=pilot.PARENT_SHA,
                    models={n: dict(weights=str(p), expected_sha256=digests[n]) for n, p in paths.items()},
                    source_experiment_meta_sha256=sha256(meta_path))
        comparison = folder / 'comparison_complete.json'
        self.dump(comparison, dict(plan=plan, models={n: dict(weights_sha256=h) for n, h in digests.items()}))
        return comparison, paths, meta_path, updates, digests['v13_best']

    def test_gpu_run_requires_a_fixed_completed_comparison(self):
        with self.assertRaisesRegex(ValueError, 'completed scratch comparison'):
            pilot.completed_comparison(None, None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'comparison_partial.json'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'completed scratch comparison'):
                pilot.completed_comparison(path, '0' * 64)

    def test_partial_schedule_or_audit_is_rejected(self):
        for problem in ('99_updates', '99_config_epochs'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                comparison, paths, meta_path, updates, start_sha = self.fixture(Path(temp))
                meta = json.loads(meta_path.read_text())
                if problem == '99_updates':
                    self.dump(updates, [{'epoch': i + 1} for i in range(99)])
                    meta['completed_audits'][updates.name] = sha256(updates)
                else:
                    meta['config']['epochs'] = 99
                self.dump(meta_path, meta)
                result = json.loads(comparison.read_text())
                result['plan']['source_experiment_meta_sha256'] = sha256(meta_path)
                self.dump(comparison, result)
                with patch.object(pilot, 'START_SHA', start_sha), self.assertRaisesRegex(ValueError, 'not completed|Incomplete'):
                    pilot.completed_comparison(comparison, sha256(comparison))

    def test_completed_identity_survives_only_unchanged_weights_and_audits(self):
        for changed in ('weight', 'audit'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                comparison, paths, meta_path, updates, start_sha = self.fixture(Path(temp))
                with patch.object(pilot, 'START_SHA', start_sha):
                    checked = pilot.completed_comparison(comparison, sha256(comparison))
                    self.assertEqual(checked['scratch_experiment_meta_sha256'], sha256(meta_path))
                    target = paths['scratch_v14_last'] if changed == 'weight' else updates
                    target.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'identity changed|audit changed'):
                        pilot.completed_comparison(comparison, sha256(comparison))


if __name__ == '__main__':
    unittest.main()
