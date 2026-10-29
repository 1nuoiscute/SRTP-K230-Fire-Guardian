import hashlib, json, sys
from pathlib import Path
import cv2
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'model_work/src'))
from audit_video_probe_overlap import fingerprint
base=ROOT/'model_work/out/cooking_vlog_intake_20261029/burner_source_review'
video=ROOT/'model_work/data/cooking_vlog_quarantine_20261029/original.webm'
data=Path('D:/SRTP_Datasets/fire_kitchen_source_v20_20261003')
manifest=data/'build_manifest.json'; result=base/'v20_overlap.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
if result.exists(): raise ValueError('refuse overwrite')
if sha(video)!='5c7150580f8665eb7e20c6bd813f072b8a7d9f87a7c73ab09d6dbabd19073204' or sha(manifest)!='15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa': raise ValueError('inputs changed')
rows=json.loads(manifest.read_text(encoding='utf8'))['lineage']
if len(rows)!=1445 or len({r['sha256'] for r in rows})!=1445: raise ValueError('unexpected lineage')
cap=cv2.VideoCapture(str(video)); selected={}; i=0
while True:
 ok,image=cap.read()
 if not ok: break
 if i in (4225,4250,4275):
  ph,pixel=fingerprint(image); selected[i]={'phash':ph,'pixel_sha256':pixel,'nearest':[],'exact':[]}
 i+=1
cap.release()
if i!=6597 or len(selected)!=3: raise ValueError('unexpected decoding')
for row in rows:
 path=data/'images/train'/row['image']
 if not path.is_file() or sha(path)!=row['sha256']: raise ValueError('lineage identity mismatch')
 image=cv2.imread(str(path))
 if image is None: raise ValueError('unreadable image')
 ph,pixel=fingerprint(image)
 for sample in selected.values():
  entry=[(ph^sample['phash']).bit_count(),row['image'],row['sha256']]
  sample['nearest']=sorted(sample['nearest']+[entry])[:3]
  if pixel==sample['pixel_sha256']: sample['exact'].append(entry)
if sha(video)!='5c7150580f8665eb7e20c6bd813f072b8a7d9f87a7c73ab09d6dbabd19073204' or sha(manifest)!='15a509075be0d2ff062894cd7de1160699bbdc7b657d8199bfba77cb80bf36fa': raise ValueError('inputs changed during audit')
summary={'role':'quarantined source overlap triage only; no predictions or independence claim','source_sha256':sha(video),'manifest_sha256':sha(manifest),'script_sha256':sha(Path(__file__)),'training_images_verified':len(rows),'frame_indices':list(selected),'frames_with_exact_matches':sum(bool(s['exact']) for s in selected.values()),'minimum_phash_distance':min(s['nearest'][0][0] for s in selected.values()),'samples':selected,'limitations':['Only three sampled frames and v20 lineage, not generic pretraining or other prediction histories.','Exact hashes and pHash cannot prove event or source independence.']}
result.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print('frames:',list(selected),'v20 images:',len(rows),'exact matches:',summary['frames_with_exact_matches'],'minimum pHash:',summary['minimum_phash_distance'])
