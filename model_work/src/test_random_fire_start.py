import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'), 'Optional Torch/Ultralytics unavailable')
class RandomStartTests(unittest.TestCase):
    def config(self):
        import ultralytics
        import yaml
        config = yaml.safe_load((Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11.yaml').read_text())
        config.update(nc=1, scale='s')
        return config

    def test_same_seed_recreates_exact_state(self):
        import torch
        from prepare_random_fire_start import random_model
        a, b = random_model(self.config(), 17), random_model(self.config(), 17)
        self.assertEqual(sum(p.numel() for p in a.parameters()), 9428179)
        self.assertEqual(a.names, {0: 'fire'})
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_seed_changes_convolutions_without_consuming_outer_rng(self):
        import torch
        from prepare_random_fire_start import random_model
        torch.manual_seed(23)
        before = torch.get_rng_state().clone()
        a, b = random_model(self.config(), 17), random_model(self.config(), 18)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        keys = [k for k, v in a.state_dict().items() if v.ndim == 4 and k.endswith('.weight') and '.dfl.' not in k]
        self.assertTrue(all(not torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in keys))

    def test_trainer_rejects_supplied_weights_or_pretraining(self):
        from train_fire_from_scratch_v14 import RandomOnlyTrainer
        trainer = object.__new__(RandomOnlyTrainer)
        trainer.args = SimpleNamespace(pretrained=False, resume=False)
        with self.assertRaisesRegex(RuntimeError, 'rejects all supplied weights'):
            trainer.get_model(weights=object())
        trainer.args.pretrained = True
        with self.assertRaisesRegex(RuntimeError, 'rejects all supplied weights'):
            trainer.get_model(weights=None)


if __name__ == '__main__':
    unittest.main()
