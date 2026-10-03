"""Acquire an explicit Commons kitchen queue into quarantine; never infer or admit labels.

Historical fetchers and frozen training sources are deliberately unchanged.
Canonical API titles and original SHA1 supplement URL-title identities.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from source_exposure import commons_source_key


def clean(value):
    text=html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
    return re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "[contact omitted]", text)


def license_allowed(name, url):
    name=clean(name); parts=urllib.parse.urlparse(clean(url))
    if parts.scheme != "https" or parts.hostname != "creativecommons.org":
        return False
    if re.fullmatch(r"CC BY(?:-SA)? (?:1\.0|2\.0|2\.5|3\.0|4\.0)", name):
        code="by-sa" if "BY-SA" in name else "by"
        return parts.path.rstrip("/")==f"/licenses/{code}/{name.split()[-1]}"
    return name=="CC0" and parts.path.rstrip("/")=="/publicdomain/zero/1.0"


_LAST_REQUEST_START=0.0
_REQUEST_INTERVAL_SECONDS=2.0


def get(url):
    global _LAST_REQUEST_START
    time.sleep(max(0.0, _REQUEST_INTERVAL_SECONDS-(time.monotonic()-_LAST_REQUEST_START)))
    _LAST_REQUEST_START=time.monotonic()
    request=urllib.request.Request(url, headers={"User-Agent":"SRTP-kitchen-source-review/1.0 (educational research)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def query(titles, width=None):
    args=dict(action="query", format="json", redirects=1, titles="|".join(titles),
              prop="imageinfo", iiprop="url|extmetadata|sha1|mime|size|metadata", iiextmetadatalanguage="en")
    if width is not None: args['iiurlwidth']=width
    raw=get("https://commons.wikimedia.org/w/api.php?"+urllib.parse.urlencode(args))
    return raw, json.loads(raw)


def known_titles(paths):
    result=set()
    def visit(value):
        if isinstance(value, dict):
            page=value.get("source_page", "")
            if page.startswith("https://commons.wikimedia.org/wiki/"):
                result.add(commons_source_key(page))
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    for path in paths:
        path=Path(path)
        if path.suffix==".csv":
            with path.open(encoding="utf-8-sig") as stream: visit(list(csv.DictReader(stream)))
        else: visit(json.loads(path.read_text(encoding="utf-8-sig")))
    return sorted(result)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def file_prefix(plan):
    prefix=plan.get("file_prefix","context3")
    if not isinstance(prefix,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,32}",prefix):
        raise ValueError("Unsafe queue filename prefix")
    return prefix


def acquire(plan_path, out):
    if out.exists(): raise ValueError("Refusing overwrite; use a new attempt directory")
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    titles=plan['titles'];prefix=file_prefix(plan)
    if len(titles)!=len(set(titles)) or any(not isinstance(t,str) or not t for t in titles):
        raise ValueError("Titles must be nonempty and unique")
    known=known_titles(plan['known_source_manifests'])
    out.mkdir(parents=True); (out/'metadata').mkdir()
    save(out/'plan_snapshot.json',plan)
    (out/'metadata'/'acquisition_source.py').write_bytes(Path(__file__).read_bytes())
    save(out/'acquisition_source_record.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_snapshot='metadata/acquisition_source.py',request_interval_seconds=_REQUEST_INTERVAL_SECONDS,no_model_predictions=True))
    prior_sha1=set(); canonical=set(known)
    # Fail closed if old source API metadata cannot be established.
    for index in range(0,len(known),40):
        raw, response=query(known[index:index+40])
        (out/'metadata'/f'known_{index:03d}.json').write_bytes(raw)
        for page in response['query']['pages'].values():
            canonical.add(page['title'])
            for info in page.get('imageinfo',[]): prior_sha1.add(info['sha1'])
    records=[]
    for index,title in enumerate(titles):
        row=dict(index=index, requested_title=title, status="quarantined_not_train_or_test",
                 acquired_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            raw, response=query(['File:'+title], plan['download_width'])
            metadata=out/'metadata'/f'candidate_{index:02d}.json';metadata.write_bytes(raw)
            pages=list(response['query']['pages'].values())
            if len(pages)!=1 or 'imageinfo' not in pages[0]: raise ValueError("Missing or ambiguous API work")
            page=pages[0];info=page['imageinfo'][0];ext=info['extmetadata']
            value=lambda key:clean(ext.get(key,{}).get('value',''))
            row.update(canonical_title=page['title'], original_sha1=info['sha1'],
                source_page='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(page['title'].replace(' ','_')),
                author=value('Artist'), license=value('LicenseShortName'), license_url=value('LicenseUrl'),
                capture_date=value('DateTimeOriginal'), credit=value('Credit'), description=value('ImageDescription'),
                original_url=info['url'], original_width=info['width'], original_height=info['height'],
                metadata_file=str(metadata.relative_to(out)), metadata_sha256=hashlib.sha256(raw).hexdigest())
            if page['title'] in canonical or info['sha1'] in prior_sha1:
                row['status']='skipped_known_or_queue_duplicate_original'
            elif not license_allowed(row['license'],row['license_url']):
                row['status']='held_license_manual_review'
            elif info['mime']!='image/jpeg':
                row['status']='held_non_jpeg'
            else:
                url=info.get('thumburl',info['url']);payload=get(url)
                with __import__('PIL.Image',fromlist=['Image']).open(io.BytesIO(payload)) as im:
                    if im.format!='JPEG': raise ValueError('Payload is not JPEG')
                    width,height=im.size;im.verify()
                filename=f'{prefix}_{index:02d}.jpg';(out/filename).write_bytes(payload)
                row.update(file=filename,sha256=hashlib.sha256(payload).hexdigest(),width=width,height=height,
                           bytes=len(payload),download_url=url)
                canonical.add(page['title']);prior_sha1.add(info['sha1'])
        except Exception as error:
            row.update(status='fetch_failed_not_admitted',error=f'{type(error).__name__}: {error}')
        records.append(row);save(out/'manifest.json',records)
        print(index,row['status'],flush=True)
    print('Quarantine only. Review license, visible targets, prior exposure and scene grouping before admission.',flush=True)
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();acquire(args.plan,args.out)
