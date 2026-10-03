"""Old-scene prediction retention on frozen features, without pseudo ground truth."""
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


def retention_terms(student, teacher, preserve, reg_max):
    """Bernoulli class KL and confidence-weighted categorical box-bin KL."""
    scores = student['scores'].float()[preserve]
    if not scores.numel():
        zero = student['scores'].sum() * 0
        return zero, zero
    target_logits = teacher['scores'].detach().float()[preserve]
    target = target_logits.sigmoid()
    entropy = F.binary_cross_entropy_with_logits(target_logits, target, reduction='none')
    cls = (F.binary_cross_entropy_with_logits(scores, target, reduction='none') - entropy).sum() / target.sum().clamp_min(1)
    confidence = target[:, 0, :]
    confidence = confidence * (confidence > .05)
    batch, anchors = confidence.shape
    boxes = student['boxes'].float()[preserve].reshape(batch, 4, reg_max, anchors)
    targets = teacher['boxes'].detach().float()[preserve].reshape(batch, 4, reg_max, anchors)
    log_p, log_q = boxes.log_softmax(2), targets.log_softmax(2)
    kl = (log_q.exp() * (log_q - log_p)).sum(2).mean(1)
    geometry = (kl * confidence).sum() / confidence.sum().clamp_min(1)
    return cls, geometry


class FireTeacherLoss(v8DetectionLoss):
    def __init__(self, model, teacher_head, preserve_paths, cls_gain=1., box_gain=1.):
        super().__init__(model)
        self.teacher = teacher_head.eval()
        self.preserve_paths = {str(Path(p).resolve()) for p in preserve_paths}
        self.cls_gain, self.box_gain = cls_gain, box_gain
        self.samples = []
        self.teacher.requires_grad_(False)

    def loss(self, preds, batch):
        total, items = super().loss(preds, batch)
        preserve = torch.tensor([str(Path(p).resolve()) in self.preserve_paths for p in batch['im_file']], device=self.device)
        with torch.no_grad():
            teacher = self.teacher.forward_head([x.detach() for x in preds['feats']], **self.teacher.one2many)
        cls, box = retention_terms(preds, teacher, preserve, self.reg_max)
        addition = torch.stack((box * self.box_gain, cls * self.cls_gain, box * 0))
        total = total + addition * preds['scores'].shape[0]
        items = items + addition.detach()
        self.samples.append((float(cls.detach()), float(box.detach()), int(preserve.sum()), int(preserve.numel())))
        return total, items
