"""Verify that an explicitly frozen backbone retains its parameters and BN buffers.

Uses the live training model, not the EMA checkpoint: EMA arithmetic and FP16
serialization can introduce rounding differences even when live weights freeze.
"""
import hashlib
import json
from pathlib import Path

import torch


def is_backbone_key(name, blocks):
    parts = name.split(".")
    return len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < blocks


def backbone_state(model, blocks):
    return {k: v.detach().cpu().contiguous().clone()
            for k, v in model.state_dict().items() if is_backbone_key(k, blocks)}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(descriptor.encode("utf-8") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FrozenBackboneAudit:
    def __init__(self, model, blocks, output):
        self.blocks = blocks
        self.output = Path(output)
        self.reference = backbone_state(model, blocks)
        self.reference_digest = state_digest(self.reference)
        self.checked_batch_epochs = set()
        self.records = []
        self.plan = {
            "blocks": blocks,
            "module_types": [type(layer).__name__ for layer in model.model[:blocks]],
            "frozen_parameter_elements": sum(p.numel() for n, p in model.named_parameters()
                                             if is_backbone_key(n, blocks)),
            "source_state_sha256": self.reference_digest,
            "source_state_tensors": len(self.reference),
            "scope": "Live training backbone parameters and all registered buffers; EMA/FP16 checkpoints are not tested for byte equality.",
        }

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps({"plan": self.plan, "checks": self.records},
                                         indent=2) + "\n", encoding="utf-8")

    def verify_state(self, trainer, stage):
        state = backbone_state(trainer.model, self.blocks)
        if set(state) != set(self.reference):
            raise RuntimeError("Frozen backbone state keys changed")
        changed = [k for k, v in state.items() if not torch.equal(v, self.reference[k])]
        if changed:
            raise RuntimeError(f"Frozen backbone tensors changed: {changed[:5]}")
        params = list(trainer.model.named_parameters())
        unexpected = [n for n, v in params if is_backbone_key(n, self.blocks) and v.requires_grad]
        trainable = sum(v.numel() for n, v in params if v.requires_grad)
        if unexpected or not trainable:
            raise RuntimeError(f"Freeze mask invalid: {unexpected[:5]}, trainable={trainable}")
        record = {"stage": stage, "epoch": getattr(trainer, "epoch", -1) + 1,
                  "backbone_state_sha256": state_digest(state),
                  "unchanged": True, "trainable_parameter_elements": trainable}
        self.records.append(record)
        self.save()
        print(f"FREEZE_AUDIT {stage}: epoch={record['epoch']}, unchanged=True, trainable={trainable}", flush=True)

    def on_start(self, trainer):
        self.verify_state(trainer, "train_start")

    def on_batch_start(self, trainer):
        # This callback runs after trainer._model_train() has frozen BN stats.
        if trainer.epoch in self.checked_batch_epochs:
            return
        active = [n for n, m in trainer.model.named_modules()
                  if is_backbone_key(n, self.blocks) and isinstance(m, torch.nn.BatchNorm2d) and m.training]
        if active:
            raise RuntimeError(f"Frozen backbone BatchNorm still training: {active[:5]}")
        self.checked_batch_epochs.add(trainer.epoch)

    def on_epoch_end(self, trainer):
        if trainer.epoch not in self.checked_batch_epochs:
            raise RuntimeError("No frozen BatchNorm mode check during this epoch")
        self.verify_state(trainer, "epoch_end")
