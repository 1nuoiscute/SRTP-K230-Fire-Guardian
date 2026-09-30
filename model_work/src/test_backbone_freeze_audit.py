"""CPU checks for the training freeze invariant; skipped when torch is absent."""
import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from backbone_freeze_audit import FrozenBackboneAudit

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm2d(2)),
                torch.nn.Linear(2, 1),
            ])


@unittest.skipIf(torch is None, "Optional PC training audit requires PyTorch; core data/rule tests remain dependency-free")
class FrozenBackboneTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model = TinyModel()
        self.audit = FrozenBackboneAudit(self.model, 1, Path(self.folder.name) / "audit.json")
        for parameter in self.model.model[0].parameters():
            parameter.requires_grad = False
        self.model.model[0][1].eval()
        self.trainer = types.SimpleNamespace(model=self.model)

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.audit.verify_state(self.trainer, "test")

    def test_unchanged_state_and_pre_epoch_start(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.audit.on_start(self.trainer)
            self.trainer.epoch = 0
            self.audit.on_batch_start(self.trainer)
            self.audit.on_epoch_end(self.trainer)
        self.assertEqual(len(self.audit.records), 2)
        self.assertEqual(self.audit.records[0]["epoch"], 0)
        self.assertTrue(all(v["unchanged"] for v in self.audit.records))

    def test_changed_parameter_rejected(self):
        self.model.model[0][0].weight.data.add_(1)
        with self.assertRaisesRegex(RuntimeError, "tensors changed"):
            self.verify()

    def test_changed_bn_running_mean_rejected(self):
        self.model.model[0][1].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "tensors changed"):
            self.verify()

    def test_changed_bn_counter_rejected(self):
        self.model.model[0][1].num_batches_tracked.add_(1)
        with self.assertRaisesRegex(RuntimeError, "tensors changed"):
            self.verify()

    def test_trainable_backbone_parameter_rejected(self):
        self.model.model[0][0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Freeze mask invalid"):
            self.verify()

    def test_bn_train_mode_rejected(self):
        self.trainer.epoch = 0
        self.model.model[0][1].train()
        with self.assertRaisesRegex(RuntimeError, "BatchNorm still training"):
            self.audit.on_batch_start(self.trainer)

    def test_no_trainable_head_rejected(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        with self.assertRaisesRegex(RuntimeError, "Freeze mask invalid"):
            self.verify()

    def test_epoch_without_bn_check_rejected(self):
        self.trainer.epoch = 0
        with self.assertRaisesRegex(RuntimeError, "No frozen BatchNorm mode check"):
            self.audit.on_epoch_end(self.trainer)


if __name__ == "__main__":
    unittest.main()
