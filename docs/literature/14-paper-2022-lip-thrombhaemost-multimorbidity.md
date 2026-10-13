# Paper Summary: Lip et al. (Thrombosis and Haemostasis 2021)

**Tiêu đề**: Improving Stroke Risk Prediction in the General Population: A Comparative Assessment of Common Clinical Rules, a New Multimorbid Index, and Machine-Learning-Based Algorithms  
**Tác giả**: Lip, Gregory Y. H.; Genaidy, Ash; Tran, George; Marroquin, Patricia; Estes, Cara; Sloop, Sue  
**Tạp chí**: *Thrombosis and Haemostasis* (2021; 122(01): 142–150) — DOI: https://doi.org/10.1055/a-1467-2993  
**Nhóm phân loại**: Dữ liệu EMR lâm sàng lớn  
**Citekey**: `Lip2022Multimorbidity` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Abstract only (no open full text)

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 3,435,224 bệnh nhân (cohort tiến cứu Mỹ, 2 năm).
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: cohort tiến cứu từ CSDL y tế.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Bệnh đồng mắc (động), nhân khẩu học.
- Xử lý dữ liệu thiếu / mất cân bằng: Không báo cáo trong phần đã trích xuất.
- **Chia dữ liệu & rò rỉ:** tập train và external validation → rủi ro **thấp** (mức Abstract).

## 3. Thuật toán & Kết quả thực nghiệm
- Gradient boosting, neural network, logistic regression; so với CHADS2, CHA2DS2-VASc, multimorbid index.
- Validation: ML c index 0.866; CHADS2 0.812; CHA2DS2-VASc 0.809; decision curve ủng hộ ML.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: không nêu.
- Điểm mạnh: quy mô rất lớn, có hiệu chuẩn và decision curve analysis.
- Hạn chế: các cách tiếp cận ML không khác biệt nhau; chỉ trích ở mức Abstract (chưa có PDF).

## Trích dẫn nguyên văn (Ground truth)
> "We studied a prospective U.S. cohort of 3,435,224 patients" (Abstract); "validation for logistic regression: c index 0.866" vs "CHADS2: c index 0.812 ... CHA2DS2-VASc: c index 0.809" (Abstract); "Decision curve analysis demonstrated that clinical utility for the ML-based formulation was better" (Abstract)
