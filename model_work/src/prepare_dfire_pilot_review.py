"""Select and render bounded, unapproved two-class development candidates.

No truth is inferred from pHash, source names or detector output. Selected raw
train WEB images exclude every cross-split component and known-similarity lead,
including unconfirmed leads, with at most one image per remaining component.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from data_integrity import sha256


PROFILE = (((),16), ((1,),16), ((0,),32), ((0,1),32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-dir',type=Path,required=True)
    parser.add_argument('--perceptual-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise SystemExit('Refusing overwrite')
    invpath = args.inventory_dir/'inventory.json'
    summary = json.loads((args.perceptual_dir/'summary.json').read_text(encoding='utf-8'))
    if sha256(invpath) != summary['inventory_sha256']: raise SystemExit('Inventory identity changed')
    for name in ('components.json','known_similarity_pairs.json'):
        if sha256(args.perceptual_dir/name) != summary[name+'_sha256']: raise SystemExit('Similarity identity changed')
    records = json.loads(invpath.read_text(encoding='utf-8'))
    components = json.loads((args.perceptual_dir/'components.json').read_text(encoding='utf-8'))
    leads = json.loads((args.perceptual_dir/'known_similarity_pairs.json').read_text(encoding='utf-8'))
    blocked = {row['candidate'] for row in leads}
    family = {}
    for component in components:
        key = hashlib.sha256(''.join(sorted(records[i]['sha256'] for i in component['members'])).encode()).hexdigest()
        for i in component['members']: family[i] = key
        if len(component['split_counts']) > 1: blocked.update(component['members'])
        if any(i in blocked for i in component['members']): blocked.update(component['members'])
    exclusions = Counter(); pools = {classes:[] for classes,_ in PROFILE}
    for i,row in enumerate(records):
        if row['split'] != 'train': exclusions['not_source_train'] += 1; continue
        if not row['file'].startswith('WEB'): exclusions['non_WEB_camera_collection'] += 1; continue
        if i in blocked: exclusions['similarity_or_known_exposure_quarantine'] += 1; continue
        if row['errors']: exclusions['structural_error'] += 1; continue
        if len(row['boxes']) > 6: exclusions['over_six_source_boxes'] += 1; continue
        # A deliberately limited first pilot, not weak/distant-smoke acceptance.
        scale = 640/max(row['width'],row['height'])
        if any(min(b[3]*row['width'],b[4]*row['height'])*scale < 24 for b in row['boxes']):
            exclusions['tiny_box_at_640'] += 1; continue
        if not row.get('phash64'): raise SystemExit('Missing source hash')
        key = family.get(i,row['sha256'])
        order = hashlib.sha256(('20261002:'+row['sha256']).encode()).hexdigest()
        pools[tuple(row['classes'])].append((order,i,key))
    selected = []; used = set()
    for classes,count in PROFILE:
        accepted = []
        for _,i,key in sorted(pools[classes]):
            if key in used: continue
            used.add(key); accepted.append({**records[i],'inventory_index':i,'candidate_group':key})
            if len(accepted) == count: break
        if len(accepted) != count: raise SystemExit('Insufficient eligible groups for requested profile')
        selected.extend(accepted)
    args.out.mkdir(parents=True); sheets = []
    for page in range((len(selected)+3)//4):
        sheet = Image.new('RGB',(1280,1200),'#252525')
        draw = ImageDraw.Draw(sheet)
        for slot,row in enumerate(selected[page*4:(page+1)*4]):
            row['index'] = page*4+slot+1
            row['manual_review'] = None
            image = Path(row['image']); label = Path(row['label'])
            if sha256(image)!=row['sha256'] or sha256(label)!=row['label_sha256']:
                raise SystemExit('Source image/label changed')
            im = Image.open(image).convert('RGB'); annotated = im.copy(); ad = ImageDraw.Draw(annotated)
            for cls,x,y,w,h in row['boxes']:
                box = [(x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height]
                color = '#ff55ff' if cls==0 else '#ff9933'
                ad.rectangle(box,outline=color,width=max(2,im.width//250))
                ad.text(box[:2],'Smoke' if cls==0 else 'Fire',fill=color)
            im.thumbnail((620,260)); annotated.thumbnail((620,260))
            ybase = slot*300
            draw.text((8,ybase+6),f"{row['index']:02d} RAW classes={row['classes']} size={row['width']}x{row['height']} boxes={len(row['boxes'])}",fill='white')
            draw.text((650,ybase+6),'SOURCE BOXES magenta=Smoke orange=Fire',fill='white')
            sheet.paste(im,(10,ybase+30)); sheet.paste(annotated,(650,ybase+30))
        output = args.out/f'sheet_{page+1:02d}.jpg'; sheet.save(output,quality=95)
        sheets.append({'file':output.name,'sha256':sha256(output)})
    result = {'role':'unapproved joint visible fire/smoke pilot candidates, no split or labels changed',
              'inventory_sha256':sha256(invpath),'perceptual_summary_sha256':sha256(args.perceptual_dir/'summary.json'),
              'script_sha256':sha256(Path(__file__)),'selection_seed':20261002,
              'selection_profile':[{'raw_classes':list(classes),'count':count} for classes,count in PROFILE],
              'exclusions':dict(exclusions),'sheets':sheets,'images':selected,
              'limitations':['Training development only; no independent scenes/kitchen ratio assertion.',
              'pHash conservative quarantine can remove unrelated sources and cannot exclude all related views.',
              'At most one selected image per hash component; components are not event identities.',
              'Tiny source boxes excluded only for this bounded first pilot, not declared unnecessary for real use.',
              'Every selected image remains pending semantic and whole-frame label review.']}
    (args.out/'review_pending.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'candidates':len(selected),'sheets':len(sheets),'exclusions':dict(exclusions)},indent=2))


if __name__ == '__main__': main()
