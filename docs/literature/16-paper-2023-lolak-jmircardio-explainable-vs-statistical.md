# Paper Summary: Lolak et al. (JMIR Cardio 2023)

**Tiêu đề**: Comparing Explainable Machine Learning Approaches With Traditional Statistical Methods for Evaluating Stroke Risk Models: Retrospective Cohort Study  
**Tác giả**: Lolak, Sermkiat; Attia, John; McKay, Gareth J; Thakkinstian, Ammarin  
**Tạp chí**: *JMIR Cardio* (2023; 7: e47736) — DOI: https://doi.org/10.2196/47736  
**Nhóm phân loại**: Dữ liệu EMR lâm sàng lớn  
**Citekey**: `Lolak2023Explainable` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC10413234: Abstract (Results)

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 275,247 bệnh nhân nguy cơ cao; 9,659 ca đột quỵ (3.5%).
- Thiết kế: cohort hồi cứu, Ramathibodi Hospital (Thái Lan) 2010–2020.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Rung nhĩ, tăng huyết áp, thuốc kháng tiểu cầu, HDL, tuổi…
- Dữ liệu thiếu: multiple imputation by chained equations; rời rạc hóa biến liên tục khi cần.
- **Mất cân bằng / chia dữ liệu:** Không báo cáo trong phần đã trích xuất.

## 3. Thuật toán & Kết quả thực nghiệm
- LR, Cox, Bayesian network, TAN, XGBoost, EBM.
- XGBoost: C-statistic 0.89, F1 0.80; EBM 0.87; LR 0.80.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: EBM, Bayesian network, so sánh yếu tố quan trọng giữa các mô hình.
- Hạn chế: một bệnh viện; chi tiết giao thức chia dữ liệu chưa trích được.

## Trích dẫn nguyên văn (Ground truth)
> "Out of 275,247 high-risk patients, 9659 (3.5%) experienced a stroke. XGBoost demonstrated the highest performance with a C-statistic of 0.89 and an F1-score of 0.80 followed by EBM and TAN ... LR and BN had similar C-statistics of 0.80" (Abstract)
