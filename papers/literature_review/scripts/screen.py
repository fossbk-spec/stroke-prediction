#!/usr/bin/env python3
"""Evisyn Buoc 4: sang loc Title/Abstract bang Gemini (batch 10) + phan luong HITL.

    WP=~/Desktop/Antigravity/WritingPapers
    GEMINI_API_KEY=... $WP/.venv/bin/python scripts/screen.py --evisyn-root $WP

Dung nguyen prompt (build_screening_prompt) va triage cua evisyn. Chi doi model:
`gemini-2.0-flash` ma ai_screener.py hardcode da bi Google ngung (404), nen
client duoc boc lai de goi --model. Ket qua tho luu tung batch vao
screening/raw_results.jsonl de chay tiep duoc khi bi ngat (resume).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


class _ModelOverride:
    """Boc genai.Client: thay model trong generate_content, retry khi 429/503."""

    def __init__(self, client, model, max_retries=6):
        self._client, self._model, self._retries = client, model, max_retries

    @property
    def models(self):
        return self

    def generate_content(self, *, model, **kw):
        for attempt in range(self._retries):
            try:
                return self._client.models.generate_content(model=self._model, **kw)
            except Exception as exc:  # google.genai.errors.APIError
                code = getattr(exc, "code", None)
                if code not in (429, 500, 503) or attempt == self._retries - 1:
                    raise
                time.sleep(min(10 * 2 ** attempt, 120))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evisyn-root", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="chi sang loc N bai dau (chay thu)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.evisyn_root).expanduser().resolve()))

    from google import genai
    from evisyn.screening.ai_screener import screen_article_batch, ScreeningOutputError
    from evisyn.screening.batch_screening import triage_screening_results

    criteria = json.loads((HERE / "evisyn/config/criteria.json").read_text(encoding="utf-8"))
    pool = [json.loads(line) for line in open(HERE / "data/evidence_pool.jsonl", encoding="utf-8")]
    if args.limit:
        pool = pool[: args.limit]

    out_dir = HERE / "screening"
    out_dir.mkdir(exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    done = {}
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            row = json.loads(line)
            # Batch loi (mang/503) khong tinh la xong -> lan chay sau gui lai
            if not row.get("error"):
                done[row["batch_ids"][0]] = row

    client = _ModelOverride(genai.Client(), args.model)
    batches = [pool[i:i + args.batch_size] for i in range(0, len(pool), args.batch_size)]
    with open(raw_path, "a", encoding="utf-8") as raw:
        for n, batch in enumerate(batches, start=1):
            ids = [r["id"] for r in batch]
            if ids[0] in done:
                continue
            articles = [{"id": r["id"], "title": r.get("title"), "year": r.get("year"),
                         "journal": r.get("journal"), "abstract": r.get("abstract") or ""}
                        for r in batch]
            try:
                results, error = screen_article_batch(articles, criteria, client=client), None
            except (ScreeningOutputError, Exception) as exc:
                results, error = [], f"{type(exc).__name__}: {exc}"[:300]
            row = {"batch_ids": ids, "model": args.model, "results": results, "error": error}
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
            if not error:
                done[ids[0]] = row
            print(f"batch {n}/{len(batches)}: {len(results)} results{' ERROR ' + error if error else ''}",
                  flush=True)

    # Phan luong theo dung triage cua evisyn (kiem evidence_quote nguyen van voi abstract)
    by_id = {r["id"]: r for r in pool}
    accepted, excluded, queue = [], [], []
    for row in done.values():
        abstracts = {i: by_id[i].get("abstract") or "" for i in row["batch_ids"] if i in by_id}
        a, e, q = triage_screening_results(row["results"], expected_ids=row["batch_ids"], abstracts=abstracts)
        accepted += a
        excluded += e
        queue += q

    def write(name, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "decision", "confidence_score", "title", "year", "journal", "doi",
                        "exclusion_reason", "evidence_quote", "contract_errors", "human_decision"])
            for r in rows:
                rec = by_id.get(r.get("id"), {})
                w.writerow([r.get("id"), r.get("decision"), r.get("confidence_score"), rec.get("title"),
                            rec.get("year"), rec.get("journal"), rec.get("doi"), r.get("exclusion_reason"),
                            r.get("evidence_quote"), "; ".join(r.get("contract_errors", [])), ""])

    write("auto_included.csv", accepted)
    write("auto_excluded.csv", excluded)
    write("human_review_queue.csv", queue)
    print(json.dumps({"screened": sum(len(r["batch_ids"]) for r in done.values()),
                      "pending_failed_batches": len(batches) - len(done),
                      "auto_included": len(accepted), "auto_excluded": len(excluded),
                      "human_review_queue": len(queue)}, indent=2))


if __name__ == "__main__":
    main()
