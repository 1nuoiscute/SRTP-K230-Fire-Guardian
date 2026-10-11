"""Extend a verified legacy one-class YOLO11 Detect head without losing fire rows."""


def transfer_fire_and_add_smoke(source,target):
    import torch
    old_head=source.model[-1]; new_head=target.model[-1]
    if old_head.nc!=1 or new_head.nc!=2 or getattr(old_head,'end2end',False) or getattr(new_head,'end2end',False):
        raise ValueError('Expected legacy one-class -> two-class Detect heads')
    if len(old_head.cv3)!=len(new_head.cv3): raise ValueError('Detection scales differ')
    old=source.state_dict(); new=target.state_dict()
    if set(old)!=set(new): raise ValueError('Model state keys differ')
    prefix=f'model.{len(source.model)-1}.cv3.'
    expanded=set()
    for i,(a,b) in enumerate(zip(old_head.cv3,new_head.cv3)):
        if not isinstance(a[-1],torch.nn.Conv2d) or not isinstance(b[-1],torch.nn.Conv2d):
            raise ValueError('Unexpected class output layer')
        last=len(a)-1
        if len(a)!=len(b) or a[-1].out_channels!=1 or b[-1].out_channels!=2:
            raise ValueError('Unexpected class output shape')
        expanded.update({f'{prefix}{i}.{last}.weight',f'{prefix}{i}.{last}.bias'})
    for key in old:
        if key not in expanded and (old[key].shape!=new[key].shape or old[key].dtype!=new[key].dtype):
            raise ValueError('Unexpected non-class state change')
        if key in expanded and (new[key].shape[0]!=2 or old[key].shape[0]!=1 or old[key].shape[1:]!=new[key].shape[1:]):
            raise ValueError('Unexpected expanded class shape')
    with torch.no_grad():
        for key in old:
            if key not in expanded: new[key].copy_(old[key])
            else:
                new[key][0].copy_(old[key][0])
                if key.endswith('.weight'): new[key][1].zero_()
                else: new[key][1].fill_(-6)
    # State dictionaries reference live tensors, but verify all rows explicitly.
    for key in old:
        check=new[key][0] if key in expanded else new[key]
        expected=old[key][0] if key in expanded else old[key]
        if not torch.equal(check,expected): raise ValueError('Fire state transfer not exact')
    return {'copied_state_tensors':len(old)-len(expanded),'expanded_class_tensors':len(expanded),
            'class_names':{0:'fire',1:'smoke'},'smoke_output_weight':0,'smoke_bias':-6,
            'scope':'Exact source state and fire output rows preserved; smoke uniformly low before training.'}
