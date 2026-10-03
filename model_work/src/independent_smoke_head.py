"""Inference-only independent fire/smoke boxes on a single shared feature network."""
import torch
from ultralytics.nn.modules.head import Detect


def combine_single_class_outputs(fire, smoke):
    if fire.ndim!=3 or smoke.ndim!=3 or fire.shape[1]!=5 or smoke.shape[1]!=5 or fire.shape[0]!=smoke.shape[0]:
        raise ValueError('Expected two [batch,5,anchors] legacy single-class outputs')
    fire_rows=torch.cat((fire,torch.zeros_like(fire[:,4:5])),dim=1)
    smoke_rows=torch.cat((smoke[:,:4],torch.zeros_like(smoke[:,4:5]),smoke[:,4:5]),dim=1)
    return torch.cat((fire_rows,smoke_rows),dim=2)


class IndependentSmokeDetect(Detect):
    """Keep fire-head boxes/scores separate; standard class-aware NMS can consume both."""
    def __init__(self, fire, smoke):
        torch.nn.Module.__init__(self)
        if fire.nc!=1 or smoke.nc!=1 or fire.end2end or smoke.end2end:
            raise ValueError('Expected two legacy single-class Detect heads')
        if fire.nl!=smoke.nl or fire.reg_max!=smoke.reg_max or not torch.equal(fire.stride,smoke.stride):
            raise ValueError('Head scale/stride configuration differs')
        self.fire_head=fire; self.smoke_head=smoke
        self.nc=2; self.nl=fire.nl; self.reg_max=fire.reg_max; self.no=4*self.reg_max+2
        self.stride=fire.stride.clone(); self._end2end=False; self.xyxy=fire.xyxy
        self.i=fire.i; self.f=fire.f; self.type=type(self).__name__
        self.np=sum(p.numel() for p in self.parameters())

    def _apply(self, fn, recurse=True):
        super()._apply(fn,recurse=recurse)
        # Detect caches/stride are plain tensors. BaseModel only handles its
        # outermost head; move both nested heads explicitly and rebuild caches.
        for head in (self.fire_head,self.smoke_head):
            for name in ('stride','anchors','strides'):
                value=getattr(head,name,None)
                if isinstance(value,torch.Tensor): setattr(head,name,fn(value))
            head.shape=None
        return self

    def forward(self, x):
        if self.training:
            raise RuntimeError('Combined model is inference-only; train the standalone smoke head')
        fire=self.fire_head(x); smoke=self.smoke_head(x)
        a=fire if isinstance(fire,torch.Tensor) else fire[0]
        b=smoke if isinstance(smoke,torch.Tensor) else smoke[0]
        combined=combine_single_class_outputs(a,b)
        return combined if self.export else (combined,{'fire':fire,'smoke':smoke})
