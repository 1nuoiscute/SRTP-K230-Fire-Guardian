"""Fetch a small new source-review queue; never admits it to training/testing."""
import argparse,hashlib,html,json,re,urllib.parse,urllib.request
from pathlib import Path
SOURCES=['Gas cooker blue flame.jpg','Gas stove blue flames.jpg','Gas stove burner flame.jpg']

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SRTP-K230-research/0.2 (educational source review)'})
    with urllib.request.urlopen(req,timeout=25) as response:return response.read()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise SystemExit('Refusing overwrite')
    args.out.mkdir(parents=True);rows=[]
    for i,title in enumerate(SOURCES):
        row={'title':title,'source_page':'https://commons.wikimedia.org/wiki/File:'+urllib.parse.quote(title.replace(' ','_')),
             'status':'quarantined_source_review_only_not_train_or_test'}
        try:
            query=urllib.parse.urlencode({'action':'query','format':'json','titles':'File:'+title,'prop':'imageinfo',
                                          'iiprop':'url|extmetadata|sha1','iiurlwidth':1280})
            meta=json.loads(get('https://commons.wikimedia.org/w/api.php?'+query))
            info=next(iter(meta['query']['pages'].values()))['imageinfo'][0]
            ext=info['extmetadata']
            def clean(name):return html.unescape(re.sub('<[^>]+>','',ext.get(name,{}).get('value',''))).strip()
            license_name=clean('LicenseShortName')
            if not any(k in license_name for k in ['CC BY','CC0','Public domain']):raise ValueError('License requires manual review')
            url=info.get('thumburl',info['url']);payload=get(url)
            path=args.out/f'candidate_{i}.jpg';path.write_bytes(payload)
            row.update({'file':path.name,'sha256':hashlib.sha256(payload).hexdigest(),'author':clean('Artist'),
                        'license':license_name,'license_url':clean('LicenseUrl'),'download_url':url,
                        'original_sha1':info['sha1'],'bytes':len(payload)})
        except Exception as error:row.update({'status':'fetch_failed_not_admitted','error':str(error)})
        rows.append(row);(args.out/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(title,row['status'],flush=True)
    print('Candidates remain quarantined: check visual relevance, labels and initialization-chain overlap before use.',flush=True)

if __name__=='__main__':main()
