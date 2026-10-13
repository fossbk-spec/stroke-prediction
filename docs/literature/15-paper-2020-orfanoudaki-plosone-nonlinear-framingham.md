# Paper Summary: Orfanoudaki et al. (PLOS ONE 2020)

**Tiêu đề**: Machine learning provides evidence that stroke risk is not linear: The non-linear Framingham stroke risk score  
**Tác giả**: Orfanoudaki, Agni; Chesley, Emma; Cadisch, Christian; Stein, Barry; Nouh, Amre; Alberts, Mark J.; Bertsimas, Dimitris  
**Tạp chí**: *PLOS ONE* (2020; 15(5): e0232414) — DOI: https://doi.org/10.1371/journal.pone.0232414  
**Nhóm phân loại**: Dữ liệu cohort  
**Citekey**: `Orfanoudaki2020NonLinear` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC7241753: Abstract; Derivation cohort para 1-2; Validation cohort para 1

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- Derivation: 4,385 người (18,793 lượt khám), Framingham Offspring; validation: 17,527 bệnh nhân Boston Medical Center.
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: cohort + kiểm định ngoài.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Biến của Revised Framingham Stroke Risk Score, ECG (sóng T), hematocrit.
- Xử lý dữ liệu thiếu / mất cân bằng: Không báo cáo trong phần đã trích xuất.
- **Chia dữ liệu & rò rỉ:** kiểm định ngoài ở cohort khác → rủi ro **thấp**; dữ liệu nhiều lượt khám/người cho thấy vì sao cần chia theo bệnh nhân.

## 3. Thuật toán & Kết quả thực nghiệm
- Optimal Classification Trees (Non-Linear Stroke Risk Score).
- C-statistic: train 87.43%, validation 75.29% so với R-FSRS 65.93%.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: cây quyết định diễn giải được, 23 hồ sơ nguy cơ.
- Điểm mạnh: so sánh trực tiếp với thang điểm Framingham.
- Hạn chế: hiệu năng giảm mạnh khi kiểm định ngoài.

## Trích dẫn nguyên văn (Ground truth)
> "inclusion of 4,385 unique participants, which translated in 18,793 distinct visits" (Derivation cohort para 2); "validation 75.29% (CI 0.74-0.76) vs 65.93% (CI 0.64-0.67)" (Abstract); "validated in a prospective multiethnic cohort of 17,527 patients" (Validation cohort para 1)
