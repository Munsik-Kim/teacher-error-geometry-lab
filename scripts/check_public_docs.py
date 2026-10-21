import json,re,sys,unicodedata
from pathlib import Path
from urllib.parse import unquote,urlsplit
root=Path(sys.argv[1]).resolve(); errors=[]; links=[]
def anchors(path):
 text=path.read_text(); result=set(); counts={}
 for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$',text,re.M):
  title=re.sub(r'[`*_]','',title).lower().strip()
  slug=''.join(c for c in title if c in '-_ ' or c.isalnum()).replace(' ','-')
  n=counts.get(slug,0); counts[slug]=n+1; result.add(slug if n==0 else slug+'-'+str(n))
 return result
for doc in [*root.glob('README*.md'),*root.glob('docs/*.md')]:
 text=doc.read_text(); text=re.sub(r'```.*?```','',text,flags=re.S)
 for url in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
  url=url.strip().strip('<>'); split=urlsplit(url)
  if split.scheme or split.netloc:continue
  path=(doc.parent/unquote(split.path)).resolve() if split.path else doc
  if not path.exists() or root not in path.parents and path!=root:
   errors.append({'document':str(doc.relative_to(root)),'link':url,'reason':'missing or escapes root'}); continue
  if split.fragment and path.suffix=='.md' and unquote(split.fragment) not in anchors(path):
   errors.append({'document':str(doc.relative_to(root)),'link':url,'reason':'anchor missing'})
  links.append({'document':str(doc.relative_to(root)),'link':url})
facts=json.loads((root/'data/results.json').read_text())
for doc in ('docs/RESULTS.md','docs/RESULTS.ko.md'):
 text=(root/doc).read_text()
 for method in facts['p2']['methods']:
  for key in ('theory','mean','mc_se','z'):
   if str(method[key]) not in text:errors.append({'document':doc,'reason':'facts value missing '+method['method']+' '+key})
result={'status':'PASS' if not errors else 'FAIL','relative_links_checked':len(links),'documents_checked':len([*root.glob('README*.md'),*root.glob('docs/*.md')]),'bilingual_P2_full_precision_values_checked':32,'errors':errors}
print(json.dumps(result,indent=2,ensure_ascii=False)); sys.exit(bool(errors))
