"""Audit decoded fixed-rate probe frames against available training/diagnostic images.
Per-image identities stay local; pHash only supplies visual-review leads.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
import cv2
import numpy as np
from data_integrity import sha256
from source_exposure import commons_source_key,training_source_keys
ROOT=Path(__file__).resolve().parents[2]

def fingerprint(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    coeff=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()
    bits=coeff>np.median(coeff[1:]);bits[0]=False
    perceptual=int(''.join('1' if v else '0' for v in bits),2)
    pixels=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    return perceptual,pixels

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--train-data',type=Path,nargs='+',required=True)
    ap.add_argument('--comparison-manifests',type=Path,nargs='+',required=True)
    ap.add_argument('--train-derived',type=Path,nargs='+',default=[])
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise SystemExit('Refusing overwrite')
    a.probe=a.probe.resolve();plan_path=a.probe/'frozen_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'));receipt=json.loads((a.probe/'acquisition.json').read_text(encoding='utf-8'))
    if sha256(plan_path)!=receipt['plan_sha256']:raise SystemExit('Frozen plan changed')
    candidates=[];clips=[];source_keys=set()
    for p in a.comparison_manifests:
        source_keys.update(training_source_keys(json.loads(p.read_text(encoding='utf-8'))))
    known={};missing=[]
    for data in a.train_data:
        for row in csv.DictReader((data/'manifest.csv').open(encoding='utf-8-sig')):
            if row['split']!='train':continue
            image=data/'images/train'/row.get('file',row.get('out_file',''))
            if row['sha256'] in known:continue
            if not image.is_file():missing.append(str(image));continue
            known[row['sha256']]=image
    # Include fine-tuning frames as well as the older CSV-backed image chain.
    derived_manifests=[]
    for manifest in [*a.comparison_manifests,*a.train_derived]:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        derived_manifests.append({'manifest_sha256':sha256(manifest),'lineage_rows':len(data.get('lineage',[])),'sample_rows':len(data.get('samples',[]))})
        for row in data.get('lineage',[]):
            digest=row.get('sha256',row.get('image_sha256'))
            if not digest:raise SystemExit('Unbound lineage image')
            if digest in known:continue
            image=manifest.parent/'images/train'/row['image']
            if not image.is_file():missing.append(str(image));continue
            known[digest]=image
        for row in data.get('samples',[]):
            digest=row['sha256']
            if digest in known:continue
            image=Path(row['source'])
            if not image.is_file():missing.append(str(image));continue
            known[digest]=image
    diagnostics=[]
    for folder in [ROOT/'model_work/data/commons_blue_review_20260927',ROOT/'model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates']:
        for row in csv.DictReader((folder/'manifest.csv').open(encoding='utf-8-sig')):
            source_keys.add(commons_source_key(row['source_page']))
            diagnostics.append((row['sha256'],folder/(row['slug']+'.jpg')))
    for row in json.loads((ROOT/'model_work/data/blue_source_queue2_20260930/manifest.json').read_text(encoding='utf-8')):
        if 'file' in row:
            source_keys.add(commons_source_key(row['source_page']))
            diagnostics.append((row['sha256'],ROOT/'model_work/data/blue_source_queue2_20260930'/row['file']))
    for row in receipt['sources']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',row['id']):raise SystemExit('Unsafe clip identity')
        path=(a.probe/row['file']).resolve()
        if not path.is_relative_to(a.probe) or sha256(path)!=row['sha256']:raise SystemExit('Original video changed or path escaped')
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);nominal=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not 0<fps<=240:raise SystemExit('Bad video metadata')
        step=max(1,round(fps/plan['configuration']['target_sample_fps']));index=0;start=len(candidates)
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index%step==0:
                ph,pixel=fingerprint(frame)
                candidates.append({'clip':row['id'],'frame_index':index,'time_seconds':index/fps,'phash':ph,'pixel_sha256':pixel,'nearest_train':[],'nearest_diagnostic':[],'exact_pixel_matches':[]})
            index+=1
        cap.release()
        if index==0:raise SystemExit('No decoded frames')
        if sha256(path)!=row['sha256']:raise SystemExit('Video changed during decoding')
        clips.append({'clip':row['id'],'source_page':row['source_page'],'source_sha256':row['sha256'],'fps':fps,'step':step,'actual_sample_fps':fps/step,'nominal_frames':nominal,'decoded_frames':index,'sampled_frames':len(candidates)-start,'original_work_seen_in_available_records':commons_source_key(row['source_page']) in source_keys})
    for domain,images in [('train',list(known.items())),('diagnostic',diagnostics)]:
        for index,(digest,path) in enumerate(images):
            if sha256(path)!=digest:raise SystemExit('Known image changed')
            image=cv2.imread(str(path))
            if image is None:raise SystemExit('Known image unreadable')
            ph,pixel=fingerprint(image)
            for frame in candidates:
                key='nearest_'+domain;entry=((frame['phash']^ph).bit_count(),str(path),digest)
                frame[key]=sorted(frame[key]+[entry])[:3]
                if pixel==frame['pixel_sha256']:frame['exact_pixel_matches'].append({'domain':domain,'path':str(path),'sha256':digest})
            if index%1000==0:print(domain,index,len(images),flush=True)
    a.out.mkdir(parents=True)
    result={'role':'available-chain duplicate/similarity triage; not proof of scene independence','plan_sha256':receipt['plan_sha256'],'derived_manifest_identities':derived_manifests,'training_images_verified':len(known),'diagnostic_images_verified':len(diagnostics),'missing_training_files':missing,'clips':clips,'frames':candidates,'limitations':['unavailable earlier training and generic pretraining not checked','decoded pixel matches are strict; lossy resizing/cropping may evade them','pHash nearest distances need visual review','no inference or training admission performed']}
    dump(a.out/'local_overlap.json',result)
    aggregate={k:v for k,v in result.items() if k not in ['frames','missing_training_files']};aggregate['missing_training_files_count']=len(missing)
    for clip in aggregate['clips']:
        ff=[f for f in candidates if f['clip']==clip['clip']]
        clip['frames_with_exact_pixel_match']=sum(bool(f['exact_pixel_matches']) for f in ff)
        clip['minimum_train_phash_distance']=min(f['nearest_train'][0][0] for f in ff)
        clip['minimum_diagnostic_phash_distance']=min(f['nearest_diagnostic'][0][0] for f in ff)
    dump(a.out/'aggregate.json',aggregate)
    print(json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
