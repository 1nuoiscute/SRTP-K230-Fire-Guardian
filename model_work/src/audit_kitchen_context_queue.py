"""Audit a quarantined source queue against explicit image inventories before prediction.

Byte/pixel equality detects copies; pHash only orders human review leads.
This does not certify unseen pretraining data or scene independence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from data_integrity import sha256


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def decoded_identity(path):
    import cv2
    import numpy as np
    # fromfile/imdecode supports Unicode Windows paths.
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Unreadable image: {path.name}')
    pixel=hashlib.sha256(str(im.shape).encode('ascii')+im.tobytes()).hexdigest()
    small=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    values=cv2.dct(cv2.resize(small,(32,32)).astype(np.float32))[:8,:8].flatten()
    bits=values>np.median(values[1:]);bits[0]=False
    perceptual=int(''.join('1' if bit else '0' for bit in bits),2)
    return pixel,perceptual


def image_paths(folder):
    if not folder.is_dir(): raise ValueError(f'Missing inventory root: {folder}')
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})


def audit(queue,plan_path,out):
    if out.exists(): raise ValueError('Refusing overwrite')
    plan=load(plan_path); candidates=[r for r in load(queue/'manifest.json') if 'file' in r]
    candidate_values={}
    for row in candidates:
        if sha256(queue/row['file'])!=row['sha256']: raise ValueError('Candidate bytes changed')
        candidate_values[row['file']]=decoded_identity(queue/row['file'])
    references={};inventory=[];roots=[]
    out.mkdir(parents=True)
    for item in plan['roots']:
        folder=Path(item['path']); paths=image_paths(folder)
        excluded=[Path(p).resolve() for p in item.get('exclude',[])]
        count=0
        for path in paths:
            if any(path.resolve().is_relative_to(p) for p in excluded): continue
            digest=sha256(path)
            inventory.append(dict(path=str(path),sha256=digest,role=item['role']))
            if digest not in references:
                pixel,perceptual=decoded_identity(path)
                references[digest]=dict(path=str(path),sha256=digest,pixel_sha256=pixel,phash=perceptual,role=item['role'])
            count+=1
        roots.append(dict(path=str(folder),role=item['role'],files=count))
        print(item['role'],folder.name,'files',count,'unique cumulative',len(references),flush=True)
    # Verify original per-file CSV identities where supplied, including val/test.
    checked=0
    for name in plan['csv_manifests']:
        path=Path(name)
        with path.open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                filename=row.get('file',row.get('out_file',''))
                image=path.parent/'images'/row['split']/filename
                if not image.is_file() or sha256(image)!=row['sha256']:
                    raise ValueError(f'Manifest identity mismatch: {image}')
                checked+=1
    values=list(references.values()); report=[]
    for row in candidates:
        pixel,perceptual=candidate_values[row['file']]
        near=sorted(values,key=lambda r:((r['phash']^perceptual).bit_count(),r['path']))[:3]
        report.append(dict(file=row['file'],sha256=row['sha256'],pixel_sha256=pixel,
            exact_byte_matches=[r['path'] for r in values if r['sha256']==row['sha256']],
            exact_pixel_matches=[r['path'] for r in values if r['pixel_sha256']==pixel],
            nearest=[dict(path=r['path'],sha256=r['sha256'],role=r['role'],distance=(r['phash']^perceptual).bit_count()) for r in near]))
    pairs=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            ap,ah=candidate_values[a['file']];bp,bh=candidate_values[b['file']]
            pairs.append(dict(a=a['file'],b=b['file'],same_byte=a['sha256']==b['sha256'],same_pixels=ap==bp,distance=(ah^bh).bit_count()))
    (out/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    result=dict(role='explicit_inventory_overlap_triage_no_model_predictions',plan_sha256=sha256(plan_path),
        queue_manifest_sha256=sha256(queue/'manifest.json'),script_sha256=sha256(Path(__file__)),
        inventory_sha256=sha256(out/'inventory.json'),roots=roots,paths_scanned=len(inventory),
        unique_byte_images=len(values),csv_image_identities_verified=checked,candidates=report,pairs=pairs,
        limits=['Only the explicitly inventoried available files were scanned.','pHash distance is a lead, not a duplicate or scene-independence verdict.','Original-source aliases are checked separately through canonical API title/SHA1 metadata.','Unrecorded data and generic pretrained exposures are unknown.'])
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    gallery(queue,report,out)
    print('Completed:',len(inventory),'paths;',len(values),'unique byte images;',checked,'manifest entries',flush=True)
    return result


def gallery(queue,rows,out):
    from PIL import Image,ImageDraw,ImageOps
    folder=out/'nearest_review';folder.mkdir()
    for row in rows:
        sheet=Image.new('RGB',(1800,780),'#202020'); draw=ImageDraw.Draw(sheet)
        paths=[queue/row['file']]+[Path(v['path']) for v in row['nearest'][:2]]
        captions=[row['file']+' candidate']+[f"nearest {i+1}: distance {v['distance']} / {v['role']}" for i,v in enumerate(row['nearest'][:2])]
        for index,(path,caption) in enumerate(zip(paths,captions)):
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((590,690))
                sheet.paste(im,(index*600+(600-im.width)//2,65+(690-im.height)//2))
            draw.text((index*600+10,15),caption,fill='white')
            draw.text((index*600+10,35),path.name[:76],fill='white')
        sheet.save(folder/(Path(row['file']).stem+'.jpg'),quality=92)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.queue,a.plan,a.out)
