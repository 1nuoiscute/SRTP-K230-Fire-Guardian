"""Observe actual optimizer.step calls, including GradScaler-skipped update gaps."""
import json
from pathlib import Path


class OptimizerCadenceAudit:
    def __init__(self, output):
        self.output = Path(output)
        self.batches = 0
        self.steps = []
        self.epochs = []
        self.optimizer = None
        self.handle = None

    def attach(self, optimizer):
        if self.handle is not None:
            raise RuntimeError('Optimizer observer already attached')
        self.optimizer = optimizer
        self.handle = optimizer.register_step_post_hook(self.on_step)

    def on_step(self, optimizer, args, kwargs):
        if optimizer is not self.optimizer:
            raise RuntimeError('Observed optimizer identity changed')
        self.steps.append({'step':len(self.steps)+1, 'batch':self.batches+1,
                           'lr':[float(group['lr']) for group in optimizer.param_groups]})

    def on_batch_end(self, trainer):
        if trainer.optimizer is not self.optimizer:
            raise RuntimeError('Trainer replaced observed optimizer')
        self.batches += 1

    def on_epoch_end(self, trainer):
        if trainer.optimizer is not self.optimizer:
            raise RuntimeError('Trainer replaced observed optimizer')
        self.epochs.append({'epoch':trainer.epoch+1, 'batches':self.batches,
                            'actual_optimizer_steps':len(self.steps)})
        self.save()

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps({'scope':'Optimizer step post-hook; batch count is separate; no inference quality claim',
                                          'batches':self.batches, 'actual_optimizer_steps':len(self.steps),
                                          'epochs':self.epochs, 'steps':self.steps}, indent=2)+'\n', encoding='utf-8')

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.save()
