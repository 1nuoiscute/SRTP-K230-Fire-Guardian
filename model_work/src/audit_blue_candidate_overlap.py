"""Exact hash and perceptual triage against available training-chain manifests.
Nearest pHash images are review leads, not proof of independence or duplication.
"""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256

def phash(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError(f'Unreadable: {path}')
    coeff=cv2.dct(cv2.resize(image,(32,32)).astype(np.float32))[:8,:8].flatten()
    median=np.median(coeff[1:]);bits=coeff>median;bits[0]=False
    return int(''.join('1' if b else '0' for b in bits),2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--initializer-data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise SystemExit('Refusing overwrite')
    inputs=json.loads((args.candidates/'manifest.json').read_text(encoding='utf-8'))
    candidates=[r for r in inputs if 'file' in r]
    candidate_hashes={r['file']:phash(args.candidates/r['file']) for r in candidates}
    nearest={r['file']:[] for r in candidates};known={};missing=[]
    for data in [args.initializer_data,args.base]:
        rows=list(csv.DictReader((data/'manifest.csv').open(encoding='utf-8-sig')))
        for row in rows:
            if row['split']!='train':continue
            image=data/'images/train'/row.get('file',row.get('out_file',''))
            if row['sha256'] in known:continue
            if not image.is_file():missing.append(str(image));continue
            known[row['sha256']]=image
    for index,(digest,image) in enumerate(known.items()):
        h=phash(image)
        for name,ch in candidate_hashes.items():
            distance=(h^ch).bit_count();nearest[name].append((distance,str(image),digest))
            nearest[name]=sorted(nearest[name])[:5]
        if index%500==0:print(f'Scanned {index}/{len(known)} training images',flush=True)
    report={'role':'candidate quarantine audit','training_images_scanned':len(known),'missing_training_files':missing,
            'limits':['only available manifest files checked','pHash is similarity triage, not scene-independent proof',
                      'no assertion about generic pretrained data or unrecorded prior data'],'candidates':[]}
    args.out.mkdir(parents=True)
    for row in candidates:
        exact=row['sha256'] in known
        report['candidates'].append({'file':row['file'],'sha256':row['sha256'],'exact_training_match':exact,
                                    'nearest_phash':nearest[row['file']],'status':'quarantined_pending_lineage_and_visual_review'})
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'},ensure_ascii=False,indent=2),flush=True)
    for row in report['candidates']:print(row['file'],'exact',row['exact_training_match'],'nearest distance',row['nearest_phash'][0][0],flush=True)

if __name__=='__main__':main()
