# Paper Summary: Bruun-Jensen et al. (BMJ Open 2024)

**Tiêu đề**: DARE-ISC model for prediction of 1-year ischaemic stroke risk in the general population and atrial fibrillation patients: a Danish nationwide cohort study  
**Tác giả**: Bruun-Jensen, Mikkel; Winther, Simon; Schmidt, Samuel Emil; Møller Dahl, Christian  
**Tạp chí**: *BMJ Open* (2024; 14(5): e076640) — DOI: https://doi.org/10.1136/bmjopen-2023-076640  
**Nhóm phân loại**: Dữ liệu EMR lâm sàng lớn  
**Citekey**: `BruunJensen2024DAREISC` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC11103190: Abstract; Methods (study population, model training); Results para 1; Strengths and limitations

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- Toàn bộ người trưởng thành Đan Mạch 2010–2017 (35 519 348 người-năm); năm kiểm định 2017: 4 580 547 người; phân nhóm rung nhĩ.
- Số ca dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: cohort sổ đăng ký quốc gia, hồi cứu.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 375 biến: chẩn đoán, nhân khẩu học, dịch vụ y tế, lối sống, yếu tố di truyền, thuốc, chỉ dấu stress; nhiều biến được tách theo độ trễ thời gian.
- Xử lý dữ liệu thiếu / mất cân bằng: Không báo cáo trong phần đã trích xuất.
- **Chia dữ liệu & rò rỉ:** train 2010–2016 (5-fold CV), tune siêu tham số riêng trên 2016, kiểm định ngoài theo thời gian trên 2017. Tuy nhiên "an individual could be in multiple cohorts" → không tách người giữa train và validation; rủi ro **trung bình**.

## 3. Thuật toán & Kết quả thực nghiệm
- Gradient boosting (LightGBM) DARE-ISC; so với GARFIELD-AF và CHA2DS2-VASc.
- AUC 0.874 trong dân số chung; rung nhĩ: AUC 0.779 vs 0.704 (GARFIELD-AF) và 0.681 (CHA2DS2-VASc).
- Average precision (PR-AUC) ở rung nhĩ: 0.119 vs 0.041 và 0.034 — bài duy nhất trong ma trận báo cáo chỉ số này.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: SHAP cho độ quan trọng tương đối của biến.
- Điểm mạnh: quy mô quốc gia, kiểm định theo thời gian, báo cáo PR-AUC phù hợp dữ liệu lệch.
- Hạn chế: 375 biến sổ đăng ký khó có ở nơi khác; cùng một người có thể xuất hiện ở cả năm train và năm kiểm định. Bài học cho N22: báo cáo AP/PR-AUC và tách người khi có nhiều bản ghi.

## Trích dẫn nguyên văn (Ground truth)
> "DARE-ISC had an AUC (95% CI) of 0.874 (0.871 to 0.876) in the general population. In AF patients, DARE-ISC was superior to the GARFIELD-AF risk model and CHA2DS2-VASc score with AUC of 0.779 ... 0.704 ... and 0.681" and "average precision scores of 0.119, 0.041 and 0.034" (Abstract); "In total, 4 580 547 individuals were included in the cohort for the validation year (2017)" (Results); "Thus, an individual could be in multiple cohorts if they fulfilled these inclusion criteria" (Methods); "trained ... on the 2010-2016 cohorts using fivefold cross-validation with external validation on the 2017 cohort" (Methods)
