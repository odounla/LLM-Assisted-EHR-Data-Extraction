"""Build a SNOMED CT lookup table for conditions from the Synthea bundles.

Codes are taken from the Condition resources in the corpus patients' own
Synthea records, so every code is traceable to the source data and none is
written by hand or by a language model. Run from the repo root:
    python scripts/build_snomed_table.py
"""
import json
import re
from collections import Counter
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
FHIR_DIR = Path("output/fhir")
OUT_PATH = Path("config/snomed_conditions.json")

SNOMED = "http://snomed.info/sct"
TAG = re.compile(r"\s*\(([a-z /]+)\)\s*$")  # trailing "(disorder)", "(finding)", ...

table = {}
counts = Counter()
missing = 0

stems = sorted(p.stem for p in CORPUS_DIR.glob("*.txt"))
for stem in stems:
    bundle_path = FHIR_DIR / f"{stem}.json"
    if not bundle_path.exists():
        missing += 1
        continue
    bundle = json.loads(bundle_path.read_text())
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        for coding in resource.get("code", {}).get("coding", []):
            if coding.get("system") != SNOMED:
                continue
            display = coding.get("display", "")
            match = TAG.search(display)
            tag = match.group(1) if match else ""
            name = TAG.sub("", display).strip().lower()
            counts[name] += 1
            table[name] = {"code": coding["code"], "display": display, "tag": tag}

for name in table:
    table[name]["count"] = counts[name]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ordered = dict(sorted(table.items(), key=lambda kv: -kv[1]["count"]))
OUT_PATH.write_text(json.dumps(ordered, indent=2))

print(f"Patients read: {len(stems) - missing} of {len(stems)} (missing bundles: {missing})")
print(f"Distinct conditions: {len(table)}. Saved to {OUT_PATH}")
print("By tag:", dict(Counter(v["tag"] for v in table.values())))
print("Top 25:")
for name, v in list(ordered.items())[:25]:
    print(f"  {v['count']:5d}  {v['code']:>12}  {name}  [{v['tag']}]")
