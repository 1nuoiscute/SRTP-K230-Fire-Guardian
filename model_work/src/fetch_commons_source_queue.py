"""Fetch explicitly listed Commons works into quarantine, rejecting already-known works."""
import argparse,csv,hashlib,html,json,re,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from source_exposure import commons_source_key
ROOT=Path(__file__).resolve().parents[2]


def get(url):
    request=urllib.request.Request(url,headers={"User-Agent":"SRTP-K230-source-review/0.3 (educational research)"})
    with urllib.request.urlopen(request,timeout=25) as response: return response.read()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--titles-json",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit("Refusing overwrite")
    titles=json.loads(args.titles_json.read_text(encoding="utf-8"))
    known=set()
    for folder in [ROOT/"model_work/data/commons_blue_review_20260927",ROOT/"model_work/data/manual_kitchen_labels_v2_20260927/commons_candidates"]:
        with (folder/"manifest.csv").open(encoding="utf-8-sig") as stream:
            known.update(commons_source_key(v["source_page"]) for v in csv.DictReader(stream))
    args.out.mkdir(parents=True)
    rows=[]
    for index,title in enumerate(titles):
        page="https://commons.wikimedia.org/wiki/File:"+urllib.parse.quote(title.replace(" ","_"))
        row={"title":title,"source_page":page,"status":"quarantined_not_train_or_test",
             "acquired_at_utc":datetime.now(timezone.utc).isoformat()}
        try:
            if commons_source_key(page) in known:
                row["status"]="skipped_already_known_original_work"
            else:
                query=urllib.parse.urlencode({"action":"query","format":"json","titles":"File:"+title,
                    "prop":"imageinfo","iiprop":"url|extmetadata|sha1|mime|size","iiurlwidth":1280})
                meta=json.loads(get("https://commons.wikimedia.org/w/api.php?"+query))
                info=next(iter(meta["query"]["pages"].values()))["imageinfo"][0]
                ext=info["extmetadata"]
                def clean(name):
                    value=html.unescape(re.sub("<[^>]+>","",ext.get(name,{}).get("value",""))).strip()
                    return re.sub(r"[\w.+-]+@[\w.-]+\.\w+","[contact omitted]",value)
                license_name=clean("LicenseShortName")
                if not any(v in license_name for v in ["CC BY","CC0","Public domain"]):
                    raise ValueError("License needs manual review before download")
                if info["mime"]!="image/jpeg": raise ValueError("Queue accepts JPEG photos only")
                payload=get(info.get("thumburl",info["url"]))
                path=args.out/f"queue2_{index:02d}.jpg";path.write_bytes(payload)
                with Image.open(path) as im: width,height=im.size;im.verify()
                row.update(file=path.name,sha256=hashlib.sha256(payload).hexdigest(),width=width,height=height,
                    author=clean("Artist"),license=license_name,license_url=clean("LicenseUrl"),
                    original_sha1=info["sha1"],description=clean("ImageDescription"),download_url=info.get("thumburl",info["url"]),bytes=len(payload))
        except Exception as error:
            row.update(status="fetch_failed_not_admitted",error=f"{type(error).__name__}: {error}")
        rows.append(row)
        (args.out/"manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(title,row["status"],flush=True)
    print("Review each photo, source group, label, license and training-chain overlap before admission.",flush=True)


if __name__=="__main__": main()
