"""Check portable documentation links and the historical bilingual P2 values."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

root = Path(sys.argv[1]).resolve()
errors = []
links = []


def anchors(path):
    result = set()
    counts = {}
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', path.read_text(), re.M):
        title = re.sub(r'[`*_]', '', title).lower().strip()
        slug = ''.join(c for c in title if c in '-_ ' or c.isalnum()).replace(' ', '-')
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(slug if number == 0 else slug + '-' + str(number))
    return result


baseline_documents = [*root.glob('README*.md'), *root.glob('docs/*.md')]
appendix = root / 'followups/linear_feature_closeout'
appendix_documents = list(appendix.glob('*.md'))
documents = sorted(set(baseline_documents + appendix_documents))
if not appendix_documents:
    errors.append({'reason': 'frozen-feature appendix documentation missing'})
for document in documents:
    text = re.sub(r'```.*?```', '', document.read_text(), flags=re.S)
    for url in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
        url = url.strip().strip('<>')
        split = urlsplit(url)
        if split.scheme or split.netloc:
            continue
        path = (document.parent / unquote(split.path)).resolve() if split.path else document
        record = {'document': str(document.relative_to(root)), 'link': url}
        if not path.exists() or root not in path.parents and path != root:
            errors.append(dict(record, reason='missing or escapes root'))
            continue
        if split.fragment and path.suffix == '.md' and unquote(split.fragment) not in anchors(path):
            errors.append(dict(record, reason='anchor missing'))
        links.append(record)

# These are the original 32 checks; their files, values and acceptance are unchanged.
facts = json.loads((root / 'data/results.json').read_text())
for document in ('docs/RESULTS.md', 'docs/RESULTS.ko.md'):
    text = (root / document).read_text()
    for method in facts['p2']['methods']:
        for key in ('theory', 'mean', 'mc_se', 'z'):
            if str(method[key]) not in text:
                errors.append({'document': document,
                               'reason': 'facts value missing ' + method['method'] + ' ' + key})

result = {'status': 'PASS' if not errors else 'FAIL',
          'relative_links_checked': len(links),
          'documents_checked': len(documents),
          'baseline_documents_checked': len(baseline_documents),
          'appendix_documents_checked': len(appendix_documents),
          'bilingual_P2_full_precision_values_checked': 32,
          'errors': errors}
print(json.dumps(result, indent=2, ensure_ascii=False))
sys.exit(bool(errors))
