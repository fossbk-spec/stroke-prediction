# Paper Summary: Lolak et al. (JMIR Cardio 2023)

**Tiêu đề**: Comparing Explainable Machine Learning Approaches With Traditional Statistical Methods for Evaluating Stroke Risk Models: Retrospective Cohort Study  
**Tác giả**: Lolak, Sermkiat; Attia, John; McKay, Gareth J; Thakkinstian, Ammarin  
**Tạp chí**: *JMIR Cardio* (2023; 7: e47736) — DOI: https://doi.org/10.2196/47736  
**Nhóm phân loại**: Dữ liệu EMR lâm sàng lớn  
**Citekey**: `Lolak2023Explainable` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC10413234: Abstract; Methods (data split); Table 2

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 275,247 bệnh nhân nguy cơ cao; 9,659 ca đột quỵ (3.5%).
- Thiết kế: cohort hồi cứu, Ramathibodi Hospital (Thái Lan) 2010–2020.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Rung nhĩ, tăng huyết áp, thuốc kháng tiểu cầu, HDL, tuổi…
- Dữ liệu thiếu: multiple imputation by chained equations; rời rạc hóa biến liên tục khi cần.
- **Mất cân bằng:** Không báo cáo trong phần đã trích xuất.
- **Chia dữ liệu & rò rỉ:** chia 80:20 theo số hồ sơ bệnh nhân, mỗi người chỉ ở một tập → rủi ro **thấp** (ví dụ thực tế của patient-level split).

## 3. Thuật toán & Kết quả thực nghiệm
- LR, Cox, Bayesian network, TAN, XGBoost, EBM.
- Bảng 2: XGBoost C-statistic 0.89, F1 0.80; EBM 0.87, F1 0.78; TAN 0.83; LR 0.80, F1 0.73.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: EBM, Bayesian network, so sánh yếu tố quan trọng giữa các mô hình.
- Hạn chế: một bệnh viện; recall không báo cáo trong phần chính (đường PR ở phụ lục).

## Trích dẫn nguyên văn (Ground truth)
> "Out of 275,247 high-risk patients, 9659 (3.5%) experienced a stroke. XGBoost demonstrated the highest performance with a C-statistic of 0.89 and an F1-score of 0.80 followed by EBM and TAN ... LR and BN had similar C-statistics of 0.80" (Abstract); "We randomly separated the data by hospital numbers into development and test sets with a ratio of 80:20; each patient appeared in only 1 data set" (Methods); Table 2: XGBoost 0.89 / F1 0.80, EBM 0.87, TAN 0.83, LR 0.80 / F1 0.73
