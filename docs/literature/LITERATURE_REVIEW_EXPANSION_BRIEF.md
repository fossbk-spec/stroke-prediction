# TASK BRIEF: MỞ RỘNG VÀ BỔ SUNG NGUỒN CHO LITERATURE REVIEW (PROJECT N22: STROKE PREDICTION)

> **Dành cho Claude / AI Agent**: Tài liệu này đóng vai trò là bản đặc tả nhiệm vụ (Task Brief & Operational Prompt) để thực hiện quy trình tìm kiếm, sàng lọc và tích hợp thêm các công trình nghiên cứu mới (2020 – 2026) vào hệ thống tổng quan tài liệu (Literature Review) của dự án **Stroke Prediction**.

---

## 1. MỤC TIÊU VÀ BỐI CẢNH DỰ ÁN

- **Chủ đề**: Dự đoán nguy cơ đột quỵ (Stroke Risk Prediction) từ dữ liệu lâm sàng/hồ sơ bệnh án điện tử dạng bảng (Tabular Clinical / EHR data).
- **Thách thức cốt lõi**:
  1. **Mất cân bằng lớp nghiêm trọng (Severe Class Imbalance)**: Tỷ lệ dương tính thường chỉ từ ~1.5% đến 5% (trong tập Kaggle chuẩn là 4.9%).
  2. **Kỷ luật không rò rỉ dữ liệu (Zero Data Leakage)**: Rất nhiều công trình trước đây mắc sai lầm nghiêm trọng khi chạy oversampling (SMOTE) hoặc chuẩn hóa trước khi chia train/test hoặc k-fold CV.
  3. **Ưu tiên độ nhạy lâm sàng (Clinical Sensitivity / Recall)**: Trong chẩn đoán y tế, hậu quả bỏ sót ca bệnh ($FN$) nghiêm trọng hơn rất nhiều so với chẩn đoán nhầm ($FP$). Ưu tiên Recall > PR-AUC > F2 > ROC-AUC (không bao giờ dùng Accuracy đơn lẻ).
  4. **Khả năng giải thích lâm sàng (Explainability / XAI)**: Tỷ số chênh (Odds Ratio) từ Logistic Regression và SHAP values cho tree ensembles.
- **Trạng thái hiện tại**: Đã có 6 bài báo hạt giống được phân tích chi tiết trong `docs/literature/` (gồm Yang 2021, Chun 2021, Dritsas 2022, Duan 2025, Akinwumi 2025, Soladoye 2025). Cần bổ sung thêm các công trình mới từ các nguồn học thuật uy tín.

---

## 2. KHUNG TIÊU CHÍ SÀNG LỌC (PICO & ELIGIBILITY CRITERIA)

Tuân thủ nghiêm ngặt cấu hình tại `docs/literature/evisyn/criteria.json`:

### 2.1. Khung PICO
- **Population (P)**: Người trưởng thành có dữ liệu yếu tố nguy cơ lâm sàng dạng bảng (demographics, comorbidities, vitals, labs, EHR).
  - *Từ khóa*: `stroke`, `cerebrovascular accident`, `ischemic stroke`, `brain infarction`.
- **Intervention (I)**: Các mô hình học máy (Machine Learning) dạng bảng (Random Forest, Gradient Boosting, XGBoost, CatBoost, LightGBM, Stacking Ensemble, chi phí nhạy cảm - Cost-Sensitive).
  - *Từ khóa*: `machine learning`, `random forest`, `gradient boosting`, `XGBoost`, `stacking`, `ensemble`.
- **Comparator (C)**: Hồi quy Logistic (Logistic Regression) hoặc các thang điểm nguy cơ truyền thống (Framingham Stroke Risk Profile - FSRP, CHA2DS2-VASc, CMCS).
  - *Từ khóa*: `logistic regression`, `Framingham Stroke Risk Profile`, `risk score`.
- **Outcome (O)**: Dự đoán biến cố đột quỵ, báo cáo đầy đủ các chỉ số đánh giá mất cân bằng (Recall/Sensitivity, PR-AUC, F1/F2, ROC-AUC, Confusion Matrix).
  - *Từ khóa*: `prediction`, `risk prediction`, `recall`, `sensitivity`, `AUC`, `PR-AUC`.

### 2.2. Tiêu chí tuyển chọn (Inclusion Criteria)
1. Dự đoán nguy cơ phát sinh đột quỵ từ dữ liệu dạng bảng/EHR (không dùng ảnh y tế CT/MRI hoặc tín hiệu EEG/ECG thuần túy).
2. Áp dụng ít nhất 01 mô hình học máy và có báo cáo cụ thể Recall / Sensitivity hoặc PR-AUC / ROC-AUC.
3. Xuất bản trong giai đoạn **2020 – 2026** trên các tạp chí bình duyệt uy tín (SJR Q1/Q2, JCR IF) hoặc kỷ yếu hội nghị top (NeurIPS, MICCAI, MLHC, IEEE JBHI, IEEE EMBC).
4. Các bài tổng quan hệ thống (Systematic Review) theo chuẩn PRISMA về ứng dụng ML trong đột quỵ.

### 2.3. Tiêu chí loại trừ (Exclusion Criteria)
1. Nghiên cứu chỉ dùng ảnh chụp não (MRI/CT), sóng điện não (EEG/ECG) hoặc dữ liệu gene/omic mà không có thông số lâm sàng dạng bảng.
2. Dự đoán kết cục sau đột quỵ (hồi phục chức năng, tử vong nội viện sau tai biến, biến chứng xuất huyết chuyển dạng) thay vì dự đoán nguy cơ khởi phát.
3. Bản thảo chưa bình duyệt (Preprints trên arXiv/medRxiv) chỉ dùng làm tài liệu tham khảo phụ, không dùng làm bằng chứng chính.
4. Bài báo bị rút lại (Retracted) hoặc không có mã định danh DOI hợp lệ.

---

## 3. CÁC NGUỒN DỮ LIỆU & CHUỖI TRUY VẤN MẪU (SEARCH STRINGS)

### 3.1. Europe PMC (REST API / Search Portal)
- **Cú pháp truy vấn chuẩn**:
```text
(TITLE_ABS:"stroke") AND (TITLE_ABS:"machine learning" OR TITLE_ABS:"random forest" OR TITLE_ABS:"gradient boosting" OR TITLE_ABS:"XGBoost" OR TITLE_ABS:"stacking" OR TITLE_ABS:"ensemble") AND (TITLE_ABS:"prediction" OR TITLE_ABS:"risk") AND (TITLE_ABS:"imbalance" OR TITLE_ABS:"SMOTE" OR TITLE_ABS:"electronic health records" OR TITLE_ABS:"risk factors" OR TITLE_ABS:"Kaggle") AND PUB_YEAR:[2020 TO 2026] AND (SRC:MED OR SRC:PMC) AND PUB_TYPE:"journal article"
```

### 3.2. Semantic Scholar (S2 API / Keyword Search)
- **Các chuỗi tìm kiếm ưu tiên**:
  1. `stroke risk prediction machine learning tabular clinical data`
  2. `stroke prediction ensemble stacking SMOTE class imbalance`
  3. `stroke prediction explainable machine learning SHAP risk factors`
  4. `stroke prediction electronic health records gradient boosting cohort`
- **Bộ lọc**: Năm từ `2020-2026`, Fields of Study: `Medicine`, `Computer Science`.
- **Ưu tiên trích xuất**: Các bài có *Highly Influential Citations* hoặc kết nối trích dẫn với các bài báo hạt giống:
  - `10.2196/30277` (Yang et al., 2021)
  - `10.1093/jamia/ocab068` (Chun et al., 2021)
  - `10.3390/s22134670` (Dritsas & Trigka, 2022)
  - `10.3389/fneur.2025.1668420` (Akinwumi et al., 2025)
  - `10.1016/j.ijmedinf.2025.106041` (Soladoye et al., 2025)

### 3.3. PubMed / MEDLINE (Entrez API)
- **Cú pháp MeSH + Text Words**:
```text
("Stroke"[Mesh] OR "Ischemic Stroke"[Mesh] OR "stroke"[Title/Abstract]) 
AND ("Machine Learning"[Mesh] OR "Supervised Machine Learning"[Mesh] OR "ensemble"[Title/Abstract] OR "XGBoost"[Title/Abstract] OR "stacking"[Title/Abstract]) 
AND ("Risk Assessment"[Mesh] OR "prediction"[Title/Abstract] OR "electronic health records"[Title/Abstract]) 
AND ("2020"[Date - Publication] : "2026"[Date - Publication])
```

### 3.4. OpenAlex (API mở 250M tác phẩm)
- Endpoint: `https://api.openalex.org/works`
- Filter tham số:
  - `concepts.id`: Các ID tương ứng với Stroke, Machine Learning, Imbalanced Data.
  - `publication_year`: `2020-2026`.
  - `type`: `article`.
  - `has_doi`: `true`.

---

## 4. BỘ CÔNG CỤ THU THẬP TỰ ĐỘNG CÓ SẴN TRONG WORKSPACE

Người dùng và Claude có thể tái sử dụng ngay các module Python có sẵn tại thư mục `C:\Antigravity\WritingPapers\evisyn\harvester\`:
- `europepmc_fetcher.py`: Gửi request trực tiếp đến Europe PMC API và trả về JSON metadata.
- `semanticscholar_fetcher.py`: Truy vấn Semantic Scholar API theo từ khóa hoặc mã DOI hạt giống.
- `pubmed_fetcher.py`: Tương tác với Entrez API của NCBI.
- `openalex_fetcher.py`: Truy vấn OpenAlex.
- `import_scopus_wos.py`: Parser nạp dữ liệu file xuất từ Scopus hoặc Web of Science.

---

## 5. ĐẶC TẢ SẢN PHẨM ĐẦU RA YÊU CẦU (DELIVERABLES)

Khi Claude thực hiện việc bổ sung tài liệu, kết quả cần được ghi thành các file cấu trúc chuẩn trong `C:\Antigravity\stroke-prediction\`:

### 5.1. File tóm tắt bài báo cá nhân (`docs/literature/{STT}-paper-{năm}-{tên_tác_giả}-{tạp_chí}.md`)
Mỗi bài báo mới được tuyển chọn cần tạo 1 file markdown theo template chuẩn:
```markdown
# Paper Summary: [Tên tác giả chính] et al. ([Tên viết tắt tạp chí] [Năm])

**Tiêu đề**: [Tên đầy đủ của bài báo]  
**Tác giả**: [Danh sách tác giả chính & Đơn vị nghiên cứu]  
**Tạp chí**: *[Tên tạp chí đầy đủ]* ([Năm]; [Tập](Số): [Trang/e-ID]) — DOI: [Link DOI]  
**Nhóm phân loại**: [Dữ liệu mở Kaggle / Dữ liệu EMR lâm sàng lớn / Tổng quan hệ thống]  

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- Quy mô mẫu ($N$), số ca đột quỵ ($y=1$), tỷ lệ mất cân bằng (positive rate).
- Thiết kế nghiên cứu (Retrospective cohort, cross-sectional, benchmark dataset).

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Số lượng và nhóm đặc trưng (Demographics, vitals, comorbidities, lab values).
- Xử lý dữ liệu khuyết thiếu (Missing value imputation) và biến đổi danh mục.
- **Kỹ thuật xử lý mất cân bằng**: (SMOTE, class_weight, undersampling, hoặc không dùng).
- **Phân chia dữ liệu & Kiểm soát rò rỉ**: (Train/Val/Test split, Patient-level grouped split, Cross-validation). Có bị SMOTE leakage không?

## 3. Thuật toán & Kết quả thực nghiệm
- Các mô hình được thử nghiệm (LR, RF, XGBoost, LightGBM, Stacking, v.v.).
- Kết quả đo lường (Ưu tiên theo thứ tự: Recall, PR-AUC, F1/F2, ROC-AUC, Specificity).
- Mô hình chiến thắng (Best model) và đóng góp kỹ thuật.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán (Critical Appraisal)
- Bài báo có dùng XAI không (SHAP, Permutation, Odds Ratio)?
- **Điểm mạnh (Strengths)** đối với bài toán thực tế.
- **Điểm yếu / Hạn chế (Limitations / Leakage risks)** cần rút kinh nghiệm cho Project N22.
```

### 5.2. Cập nhật Ma trận Tổng quan Tài liệu (Literature Matrix)
Bổ sung các dòng tương ứng vào bảng ma trận so sánh tổng hợp trong `docs/DETAILED_EXAM_REPORT.md`:
- Cột so sánh: *Tác giả & Năm*, *Quy mô mẫu ($N$)*, *Tỷ lệ dương tính*, *Thuật toán tốt nhất*, *Recall*, *ROC-AUC / PR-AUC*, *Cách xử lý mất cân bằng*, *Đánh giá rủi ro rò rỉ dữ liệu (Data Leakage Risk)*.

### 5.3. Cập nhật danh mục file tham khảo gốc
Ghi nhận mã DOI và tên file PDF quy ước vào file `data/SSoT_pdf_references/README.md` theo chuẩn:
`[Year]_[AuthorLastName]_[PaperTitle_Abbrev].pdf`.

---

## 6. HƯỚNG DẪN PROMPT THỰC THI NHANH CHO CLAUDE

Khi nạp file này vào Claude, người dùng có thể gửi kèm câu lệnh mẫu:
> *"Claude, hãy đọc kỹ bản đặc tả nhiệm vụ trong file `docs/literature/LITERATURE_REVIEW_EXPANSION_BRIEF.md`. Hãy sử dụng các chuỗi truy vấn PICO được đề xuất để tìm kiếm và sàng lọc 3–5 bài báo mới nhất (2022–2026) về Stroke Prediction sử dụng Tabular ML/EHR. Với mỗi bài báo tìm được, hãy xuất file tóm tắt chi tiết theo đúng template mục 5.1 và chỉ rõ phân tích về nguy cơ Data Leakage của bài báo đó."*
