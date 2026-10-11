"""Train only smoke output rows; audit all live fire-path state and protect EMA."""
import json
from pathlib import Path


class FirePathPreservation:
    def __init__(self, model, output):
        import torch
        self.output = Path(output)
        self.reference = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        head = model.model[-1]
        if head.nc != 2 or getattr(head, 'end2end', False):
            raise ValueError('Expected legacy two-class head')
        self.rows = set()
        for i, branch in enumerate(head.cv3):
            if not isinstance(branch[-1], torch.nn.Conv2d) or branch[-1].out_channels != 2:
                raise ValueError('Unexpected classification layer')
            prefix = f'model.{len(model.model)-1}.cv3.{i}.{len(branch)-1}'
            self.rows.update((prefix+'.weight', prefix+'.bias'))
        if not self.rows <= self.reference.keys():
            raise ValueError('Classification state names differ')
        self.handles = []
        self.checks = []
        self.batch_checks = 0
        self.plan = {'trainable_tensors':sorted(self.rows),
                     'effective_smoke_parameter_elements':sum(self.reference[k][1].numel() for k in self.rows),
                     'protected_state_tensors':len(self.reference),
                     'scope':'All live registered parameters/buffers except smoke row 1; row 0 retained; EMA protected state copied before validation/checkpoint save'}

    @staticmethod
    def mask_fire_gradient(gradient):
        result = gradient.clone()
        result[0].zero_()
        return result

    def configure(self, model, optimizer):
        if self.handles:
            raise RuntimeError('Preservation hooks already configured')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in self.rows)
            if name in self.rows:
                self.handles.append(parameter.register_hook(self.mask_fire_gradient))
        # Zero gradients alone do not stop AdamW decay of protected fire rows.
        for group in optimizer.param_groups:
            group['weight_decay'] = 0.0
        self.verify(model, 'configured')

    def set_fixed_bn(self, model):
        import torch
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.batch_checks += 1

    def verify(self, model, stage, epoch=None, check_grad=True):
        import torch
        state = model.state_dict()
        if state.keys() != self.reference.keys():
            raise RuntimeError('Fire-path state keys changed')
        for key, original in self.reference.items():
            current = state[key].detach().cpu()
            if key in self.rows:
                current, original = current[0], original[0]
            if not torch.equal(current, original):
                raise RuntimeError(f'Protected fire-path state changed: {key}')
        if check_grad and {k for k,v in model.named_parameters() if v.requires_grad} != self.rows:
            raise RuntimeError('Unexpected trainable parameter mask')
        self.checks.append({'stage':stage, 'epoch':epoch, 'unchanged':True})
        self.save()

    def sync_ema(self, model):
        import torch
        state = model.state_dict()
        if state.keys() != self.reference.keys():
            raise RuntimeError('EMA state keys changed')
        with torch.no_grad():
            for key, original in self.reference.items():
                destination = state[key][0] if key in self.rows else state[key]
                source = original[0] if key in self.rows else original
                destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
        self.verify(model, 'ema_after_protected_copy', check_grad=False)

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps({'plan':self.plan,'fixed_bn_batches':self.batch_checks,
                                          'checks':self.checks}, indent=2)+'\n', encoding='utf-8')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.save()
