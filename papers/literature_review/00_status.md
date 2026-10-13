# Literature review — trạng thái (bàn giao sang máy khác)

Cập nhật: 2026-10-13. Theo workflow `fossbk-spec/WritingPapers` → `.agents/workflows/literature-review.md` (Evisyn).

- **review_type:** RAPID REVIEW
- **Mục tiêu:** thay Ma trận Y văn trong `docs/DETAILED_EXAM_REPORT.md` (Phần II, Câu 2.1). Bảng hiện tại trích 4 nguồn (Garg 2025, Abed 2022, Shao 2021, "N22 Team 2025") không có trong `docs/literature/` và không có PDF trong `data/SSoT_pdf_references/` — cần thay bằng nguồn đã qua pipeline này.

## Tiến độ

| Bước | Trạng thái |
|---|---|
| 0. Loại review | ✅ RAPID REVIEW |
| 1. PICO + search string | ✅ **Gate 1 đã duyệt** (`evisyn/config/criteria.json`, chuỗi tìm kiếm trong `scripts/harvest.py`) |
| 2. Harvest | ✅ Europe PMC 1,011 · OpenAlex 545 · Semantic Scholar 99 · PubMed 0 (NCBI WAF chặn IP) |
| 3. Dedupe | ✅ 1,655 → 1,406 bài (`data/evidence_pool.jsonl`, số liệu `data/prisma_counts.json`) |
| 4. AI screening | ⏳ **3/141 batch xong** (`screening/raw_results.jsonl`); 138 batch lỗi mạng, cần chạy lại |
| Gate 2 | ⏳ duyệt `screening/human_review_queue.csv` |
| 5. Evidence matrix, Gate 3, 3.5 | chưa |
| 6. PRISMA, library.bib, Gate 4 | chưa |

## Chạy tiếp trên máy khác

```bash
gh repo clone fossbk-spec/WritingPapers ~/Desktop/Antigravity/WritingPapers
cd ~/Desktop/Antigravity/WritingPapers && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
WP=~/Desktop/Antigravity/WritingPapers
cd <stroke-prediction>/papers/literature_review
export GEMINI_API_KEY=...        # không commit key; chỉ đặt GEMINI_API_KEY (nếu có cả GOOGLE_API_KEY thì SDK ưu tiên GOOGLE_API_KEY)
$WP/.venv/bin/python scripts/screen.py --evisyn-root $WP --model gemini-3-flash-preview
```

`screen.py` tự resume: bỏ qua batch đã thành công, chỉ gửi lại batch lỗi/chưa chạy. Kết quả: `screening/auto_included.csv`, `auto_excluded.csv`, `human_review_queue.csv` (Gate 2: điền cột `human_decision`).

Muốn bổ sung PubMed (từ mạng không bị NCBI chặn): `NCBI_EMAIL=... $WP/.venv/bin/python scripts/harvest.py --evisyn-root $WP` — **lưu ý**: chạy lại harvest sẽ ghi đè `evidence_pool.jsonl` và đánh lại ID, nên phải xoá `screening/raw_results.jsonl` và sàng lọc lại từ đầu.

## Sai khác so với workflow gốc (ghi vào phần hạn chế)

- `evisyn/screening/ai_screener.py` hardcode `gemini-2.0-flash` — Google đã ngừng (404); các bản 3.5–3.8 flash đang quá tải (503) ngày 2026-10-13 → dùng `gemini-3-flash-preview` qua wrapper trong `scripts/screen.py` (không sửa code WritingPapers). Nên sửa model mặc định ở WritingPapers.
- PubMed bị NCBI WAF chặn → dựa vào Europe PMC (`SRC:MED` bao phủ MEDLINE).
- OpenAlex full-text search cho 46k kết quả → dùng bộ lọc `title.search:stroke` + `title_and_abstract.search:"machine learning" prediction risk` (545).
- `query_builder.py` chỉ lấy 4 từ đầu của P/I, bỏ C/O, hardcode năm 2018 → chuỗi tìm kiếm viết tay (đã duyệt Gate 1).
- Semantic Scholar không có API key → bị 429, chỉ lấy được 99 kết quả đầu.
