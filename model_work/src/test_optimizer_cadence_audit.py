import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
try:
    import torch
except ImportError:
    torch = None
from optimizer_cadence_audit import OptimizerCadenceAudit


@unittest.skipIf(torch is None, 'PyTorch unavailable in lightweight CI')
class CadenceTests(unittest.TestCase):
    def test_batches_are_not_updates_and_detach_stops_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            value = torch.nn.Parameter(torch.tensor([1.]))
            optimizer = torch.optim.SGD([value], lr=.1)
            trainer = SimpleNamespace(optimizer=optimizer, epoch=0)
            audit = OptimizerCadenceAudit(Path(directory)/'audit.json'); audit.attach(optimizer)
            audit.on_batch_end(trainer)  # accumulation/skipped batch, no step
            value.grad = torch.ones_like(value); optimizer.step()
            audit.on_batch_end(trainer); audit.on_epoch_end(trainer)
            saved = json.loads(audit.output.read_text())
            self.assertEqual((saved['batches'], saved['actual_optimizer_steps']), (2,1))
            self.assertEqual(saved['steps'][0], {'step':1, 'batch':2, 'lr':[.1]})
            audit.close(); optimizer.step(); self.assertEqual(len(audit.steps),1)

    def test_optimizer_replacement_and_duplicate_attach_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            value = torch.nn.Parameter(torch.tensor([1.]))
            optimizer = torch.optim.SGD([value], lr=.1)
            audit = OptimizerCadenceAudit(Path(directory)/'audit.json'); audit.attach(optimizer)
            with self.assertRaises(RuntimeError): audit.attach(optimizer)
            with self.assertRaises(RuntimeError): audit.on_batch_end(SimpleNamespace(optimizer=object()))
            audit.close()


if __name__ == '__main__': unittest.main()
