"""Bind manually authored decisions to the actual source bytes and viewed sheets.

This validator does not infer visual truth or independently approve labels.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from data_integrity import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pending',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite')
    pending = json.loads(a.pending.read_text(encoding='utf-8'))
    decisions = json.loads(a.decisions.read_text(encoding='utf-8'))
    if decisions['review_pending_sha256'] != sha256(a.pending): raise SystemExit('Pending identity changed')
    if decisions.get('actual_all_sheets_viewed') is not True: raise SystemExit('Actual review confirmation missing')
    for sheet in pending['sheets']:
        if sha256(a.pending.parent/sheet['file']) != sheet['sha256']: raise SystemExit('Sheet changed')
    rows = pending['images']; notes = decisions['reviews']
    if len(notes) != len(rows) or {r['index'] for r in notes} != {r['index'] for r in rows}:
        raise SystemExit('Decisions must cover every index exactly once')
    by_index = {r['index']:r for r in notes}
    for row in rows:
        if sha256(Path(row['image'])) != row['sha256'] or sha256(Path(row['label'])) != row['label_sha256']:
            raise SystemExit('Original image/label changed')
        note = by_index[row['index']]
        if note['decision'] not in ('approved_source_labels','hold') or not note.get('note'):
            raise SystemExit('Explicit decision/note missing')
        row['manual_review'] = note
        row['approved_for_training'] = note['decision']=='approved_source_labels'
        if row['approved_for_training'] and row['errors']: raise SystemExit('Structural error approved')
        if note.get('development_group'):
            row['reviewed_development_group'] = note['development_group']
        else: row['reviewed_development_group'] = row['candidate_group']
    pending.update(role='completed whole-frame/source-box visual pilot review; development labels only',
                   reviewed_utc=datetime.now(timezone.utc).isoformat(),
                   reviewer=decisions['reviewer'],policy=decisions['policy'],
                   review_pending_sha256=sha256(a.pending),decisions_sha256=sha256(a.decisions),
                   approved_images=sum(r['approved_for_training'] for r in rows),
                   held_images=sum(not r['approved_for_training'] for r in rows))
    a.out.write_text(json.dumps(pending,indent=2),encoding='utf-8')
    print(json.dumps({'approved':pending['approved_images'],'held':pending['held_images'],'complete_sha256':sha256(a.out)},indent=2))


if __name__ == '__main__': main()
