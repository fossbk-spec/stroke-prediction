#!/usr/bin/env python3
"""Evisyn Buoc 2-3: thu thap da nguon + khu trung lap cho rapid review stroke-prediction.

Chay bang venv cua WritingPapers (ha tang evisyn dung chung, khong copy code):
    WP=~/Desktop/Antigravity/WritingPapers
    $WP/.venv/bin/python scripts/harvest.py --evisyn-root $WP

Chuoi tim kiem da duoc tac gia duyet o Gate 1 (2026-10-13).
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

PUBMED_QUERY = (
    '("stroke"[tiab] AND ("risk prediction"[tiab] OR "predict*"[tiab])) '
    'AND ("machine learning"[tiab] OR "ensemble"[tiab] OR "random forest"[tiab] '
    'OR "gradient boosting"[tiab] OR "SMOTE"[tiab]) '
    'NOT ("MRI"[ti] OR "CT"[ti] OR "imaging"[ti] OR "outcome after stroke"[tiab]) '
    'AND 2020:2026[dp] AND English[la]'
)
EUROPEPMC_QUERY = (
    '(TITLE_ABS:"stroke" AND (TITLE_ABS:"risk prediction" OR TITLE_ABS:"predict*")) '
    'AND (TITLE_ABS:"machine learning" OR TITLE_ABS:"ensemble" OR TITLE_ABS:"random forest" '
    'OR TITLE_ABS:"gradient boosting" OR TITLE_ABS:"SMOTE") '
    'NOT (TITLE:"MRI" OR TITLE:"CT" OR TITLE:"imaging") '
    'AND PUB_YEAR:[2020 TO 2026] AND LANG:"eng" AND (SRC:MED OR SRC:PMC)'
)
OPENALEX_SEARCH = ""  # dung bo loc title/abstract thay cho full-text search (46k ket qua, qua rong)
OPENALEX_FILTER = (
    "from_publication_date:2020-01-01,to_publication_date:2026-12-31,has_doi:true,type:article,language:en,"
    'title.search:stroke,title_and_abstract.search:"machine learning" prediction risk'
)
S2_QUERY = "stroke risk prediction machine learning imbalanced"


def _retry(fetch, attempts=3, wait=60):
    """Semantic Scholar khong key hay tra 429 - fetcher tra [] khi bo cuoc, nen thu lai."""
    import time
    for i in range(attempts):
        recs = fetch()
        if recs:
            return recs
        if i < attempts - 1:
            time.sleep(wait)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evisyn-root", required=True)
    ap.add_argument("--max-per-source", type=int, default=1200)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.evisyn_root).expanduser().resolve()))

    from evisyn.harvester.pubmed_fetcher import fetch_pubmed
    from evisyn.harvester.europepmc_fetcher import fetch_europepmc
    from evisyn.harvester.openalex_fetcher import fetch_openalex
    from evisyn.harvester.semanticscholar_fetcher import fetch_semanticscholar
    from evisyn.processing.deduplicate import deduplicate_records

    n = args.max_per_source
    sources = {
        "pubmed": (PUBMED_QUERY, lambda: fetch_pubmed(PUBMED_QUERY, retmax=n)),
        "europepmc": (EUROPEPMC_QUERY, lambda: fetch_europepmc(EUROPEPMC_QUERY, max_results=n)),
        "openalex": (f"{OPENALEX_SEARCH} | filter={OPENALEX_FILTER}",
                     lambda: fetch_openalex(OPENALEX_SEARCH, OPENALEX_FILTER, max_results=n)),
        "s2": (S2_QUERY, lambda: _retry(lambda: fetch_semanticscholar(S2_QUERY, max_results=100))),
    }

    all_records, search_log = [], []
    for name, (query, fetch) in sources.items():
        try:
            recs = fetch()
            status = "ok"
        except Exception as exc:  # ghi lai loi nguon (vd WAF NCBI) thay vi dung ca pipeline
            recs, status = [], f"error: {exc}"[:300]
        # Loc nam 2020-2026 phong khi API khong ap dung bo loc
        recs = [r for r in recs if not r.year or 2020 <= r.year <= 2026]
        print(f"[{name}] {len(recs)} records ({status})")
        search_log.append({"database": name, "query": query, "date": date.today().isoformat(),
                           "n_results": len(recs), "status": status})
        all_records.extend(recs)

    # ID noi bo duy nhat truoc khi khu trung lap (cac fetcher co the trung ID)
    for i, r in enumerate(all_records, start=1):
        r.id = f"EVI_{i:05d}"
    unique, n_dup = deduplicate_records(all_records)

    out = HERE / "data"
    out.mkdir(exist_ok=True)
    with open(out / "evidence_pool.jsonl", "w", encoding="utf-8") as f:
        for r in unique:
            f.write(r.model_dump_json() + "\n")
    counts = {k: next(s["n_results"] for s in search_log if s["database"] == k) for k in sources}
    counts.update(total_identified=len(all_records), duplicates_removed=n_dup, unique_records=len(unique))
    (out / "search_log.json").write_text(json.dumps(search_log, indent=2, ensure_ascii=False))
    (out / "prisma_counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
