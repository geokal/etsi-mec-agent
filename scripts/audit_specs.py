"""Audit data/specs: does each file and manifest key name the spec it contains?

    uv run python scripts/audit_specs.py

The monitor used to save whatever PDF an Exa lookup returned under the spec
number it was asked for, so keys and content disagree for most of the corpus.
This reports the three ways that shows up — wrong document, unverifiable URL,
spec never fetched — and asserts the identity parser against real ETSI URLs.
"""
import collections
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from etsi_mec_agent.tools.monitor_deliver import _spec_id_from_url

# --- parser check against real ETSI deliver URL shapes ---
assert _spec_id_from_url(
    "https://www.etsi.org/deliver/etsi_gs/MEC/001_099/002/04.01.01_60/gs_MEC002v040101p.pdf") == "MEC002"
assert _spec_id_from_url(
    "https://etsi.org/deliver/etsi_gr/MEC/001_099/059/04.01.01_60/gr_mec059v040101p.pdf") == "MEC059"
assert _spec_id_from_url(
    "https://www.etsi.org/deliver/etsi_gr/MEC-DEC/001_099/025/02.01.01_60/gr_mec-dec025v020101p.pdf") == "MEC-DEC025"
try:
    _spec_id_from_url("https://www.etsi.org/deliver/etsi_gs/MEC/001_099/002/04.01.01_60/index.html")
except ValueError:
    pass
else:
    raise AssertionError("a non-PDF URL must not yield a spec ID")
print("[check] _spec_id_from_url: gs_/gr_/sub-series/non-PDF all OK\n")

import pymupdf

SPEC_DIR = pathlib.Path("data/specs")
by_hash = collections.defaultdict(list)
for f in sorted(SPEC_DIR.glob("*.pdf")):
    by_hash[hashlib.md5(f.read_bytes()).hexdigest()].append(f.name)

content_of = {}
for h, names in by_hash.items():
    doc = pymupdf.open(stream=(SPEC_DIR / names[0]).read_bytes(), filetype="pdf")
    head = " ".join(doc[0].get_text().split())[:200]
    doc.close()
    m = re.search(r"ETSI\s+(GS|GR|FGS|SRL)\s+MEC[-\s]*([A-Z]*)(\d{3,5})", head)
    v = re.search(r"V(\d+\.\d+\.\d+)", head)
    content_of[h] = (f"MEC{m.group(2)}{m.group(3)}" if m else "?", v.group(1) if v else "?")

print(f"pdfs: {sum(len(v) for v in by_hash.values())} files, {len(by_hash)} unique documents\n")

manifest = json.loads((SPEC_DIR / "manifest.json").read_text())
wrong_key, ok_key = [], 0
for key, url in sorted(manifest.items()):
    try:
        declared = _spec_id_from_url(url)
    except ValueError:
        declared = "UNVERIFIABLE"
    if declared == key:
        ok_key += 1
    else:
        wrong_key.append((key, declared, url))
print(f"manifest keys whose URL names that key: {ok_key}/{len(manifest)}")
for key, declared, url in wrong_key[:10]:
    print(f"   {key} -> URL declares {declared}  {url.rsplit('/', 1)[-1]}")
if len(wrong_key) > 10:
    print(f"   ... {len(wrong_key) - 10} more")

have = collections.Counter(content_of[h][0] for h in by_hash)
present = {n for n in have if n != "?"}
print(f"\ndistinct spec documents actually on disk: {len(present)}")
nums = sorted(int(n[3:]) for n in present if re.fullmatch(r"MEC\d{3}", n))
gaps = [i for i in range(1, (max(nums) if nums else 0) + 1) if i not in nums]
print(f"numbered range held: {nums[0]:03d}-{nums[-1]:03d}" if nums else "no numbered specs found")
print(f"numbers never fetched in that range ({len(gaps)}): "
      + ", ".join(f"MEC{i:03d}" for i in gaps[:25]))
multi = {h: names for h, names in by_hash.items() if len(names) > 1}
print(f"\nspec numbers stored under more than one filename group: "
      f"{sum(1 for h in multi if content_of[h][0] != '?')} groups")
for h, names in sorted(multi.items(), key=lambda x: -len(x[1]))[:5]:
    c, v = content_of[h]
    print(f"   {c} V{v}: {len(names)} files e.g. {', '.join(names[:4])}")
