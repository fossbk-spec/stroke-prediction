# BÁO CÁO GIẢI ĐỀ KIỂM TRA TỰ LUẬN & ĐỊNH HƯỚNG TRIỂN KHAI PROJECT N22: STROKE PREDICTION USING MACHINE LEARNING

## TỔNG QUAN DỰ ÁN
- **Tên dự án:** Project N22 - Stroke Prediction using Machine Learning
- **Mục tiêu:** Xây dựng mô hình Học máy dự đoán nguy cơ đột quỵ từ dữ liệu lâm sàng y tế, đảm bảo độ tin cậy, không rò rỉ dữ liệu (Zero Data Leakage) và có tính giải thích lâm sàng cao.
- **Bộ dữ liệu chuẩn:** Kaggle Stroke Prediction Dataset (5,110 bệnh nhân, 12 thuộc tính).

---

## PHẦN I: PROBLEM STATEMENT AND RISK-COST ANALYSIS (2.0 điểm)

### Câu 1.1: Phát biểu bài toán lâm sàng & Không gian dữ liệu
1. **Tập bệnh nhân mục tiêu (Target Population):**
   - Quần thể người trưởng thành trong cộng đồng hoặc nhóm bệnh nhân nguy cơ cao (tiền sử cao huyết áp, đái tháo đường, tim mạch) cần tầm soát, dự báo sớm nguy cơ đột quỵ trong chăm sóc sức khỏe ban đầu.
2. **Không gian đặc trưng đầu vào ($X$):**
   - $X \in \mathbb{R}^{d}$ bao gồm 11 thuộc tính lâm sàng và nhân khẩu học:
     - `id`: Mã định danh bệnh nhân.
     - `gender`: Giới tính (`Male`, `Female`, `Other`).
     - `age`: Tuổi của bệnh nhân (biến liên tục).
     - `hypertension`: Tiền sử cao huyết áp (0: Không, 1: Có).
     - `heart_disease`: Tiền sử bệnh tim (0: Không, 1: Có).
     - `ever_married`: Tình trạng kết hôn (`Yes`, `No`).
     - `work_type`: Loại hình công việc (`Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked`).
     - `Residence_type`: Loại hình nơi ở (`Urban`, `Rural`).
     - `avg_glucose_level`: Mức đường huyết trung bình (biến liên tục).
     - `bmi`: Chỉ số khối cơ thể Body Mass Index (biến liên tục).
     - `smoking_status`: Trạng thái hút thuốc (`formerly smoked`, `never smoked`, `smokes`, `Unknown`).
3. **Nhãn đầu ra ($Y$):**
   - $Y \in \{0, 1\}$ (0: Không xảy ra đột quỵ, 1: Có xảy ra đột quỵ).
4. **Phân loại bài toán Học máy:**
   - **Supervised Learning** (Học có giám sát) - **Binary Classification** (Phân loại nhị phân).

### Câu 1.2: Phân tích rủi ro bất đối xứng & Lựa chọn độ đo
1. **Phân tích rủi ro bất đối xứng giữa False Positive (FP) và False Negative (FN):**
   - Trong bài toán chẩn đoán y khoa, chi phí/rủi ro của sai sót là bất đối xứng nghiêm trọng ($Cost(FN) \gg Cost(FP)$):
     - **False Negative (FN - Âm tính giả):** Mô hình dự đoán bệnh nhân "Không có nguy cơ đột quỵ" ($Y=0$), nhưng thực tế bệnh nhân "Có nguy cơ đột quỵ" ($Y=1$). Hậu quả: Bệnh nhân bị bỏ sót chẩn đoán, không nhận được các biện pháp can thiệp dự phòng y tế kịp thời, dẫn đến biến cố đột quỵ cấp, tổn thương não vĩnh viễn, tàn phế hoặc tử vong.
     - **False Positive (FP - Dương tính giả):** Mô hình dự đoán "Có nguy cơ đột quỵ" ($Y=1$), nhưng thực tế "Không đột quỵ" ($Y=0$). Hậu quả: Bệnh nhân chịu thêm các chi phí khám tầm soát chuyên sâu, xét nghiệm bổ sung hoặc lo âu tâm lý ngắn hạn.
   - **Kết luận:** FN mang lại hậu quả y khoa và tính mạng vô cùng nghiêm trọng, do đó mục tiêu hàng đầu là tối thiểu hóa tối đa số lượng FN.

2. **Lựa chọn độ đo đánh giá (Evaluation Metrics):**
   - **Hiện tượng Class Imbalance:** Bộ dữ liệu chỉ có ~4.9% mẫu dương tính ($Y=1$). Nếu sử dụng **Accuracy (Độ chính xác toàn cục)** làm độ đo chính, một mô hình thơ ngây (Naive Predictor) luôn dự đoán $Y=0$ sẽ dễ dàng đạt Độ chính xác $95.1\%$ (Null Accuracy), nhưng hoàn toàn vô dụng trong lâm sàng vì $Recall = 0\%$.
   - **Hệ thống độ đo bắt buộc:**
     - **Sensitivity / Recall ($TP / (TP + FN)$):** Độ đo ưu tiên số 1, đo lường khả năng bắt trọn tất cả các ca có nguy cơ đột quỵ thực sự.
     - **PR-AUC (Area Under Precision-Recall Curve):** Độ đo phản ánh chính xác hiệu năng mô hình trên tập dữ liệu mất cân bằng lớp nghiêm trọng.
     - **F1-Score / $F_{\beta}$-Score ($\beta = 2$):** $F_2$-score gán trọng số cao hơn cho Recall so với Precision, giúp cân bằng giữa việc loại bỏ FN và kiểm soát FP.
     - **ROC-AUC:** Đánh giá khả năng phân tách tổng thể của mô hình ở các ngưỡng xác suất khác nhau.

---

## PHẦN II: DEEP LITERATURE REVIEW — CORE FOCUS (5.0 điểm)

> Loại tổng quan: **Rapid review** (không phải systematic review), thực hiện theo quy trình Evisyn 6 bước (PRISMA 2020, 4 chốt duyệt của con người). Toàn bộ dữ liệu gốc nằm trong `docs/literature/evisyn/`: `criteria.json` (PICO, tiêu chí, chuỗi tìm kiếm), `search_log.json`, `evidence_pool.jsonl` (734 bản ghi kèm quyết định sàng lọc), `human_review_queue.csv`, `evidence_matrix.json` (ma trận lõi kèm trích dẫn nguyên văn) và `references.bib`.

### Câu 2.1: Ma trận y văn & Tiêu chuẩn thẩm định
1. **Tiêu chuẩn lựa chọn bài báo khoa học:**
   - **Xếp hạng:** Tạp chí bình duyệt, mục tiêu SJR Q1/Q2 (Scopus / Web of Science). Xếp hạng từng tạp chí phải được tra trên scimago.com và ghi vào trường `venue_rank`; AI không tự suy đoán xếp hạng.
   - **Thời gian xuất bản:** 2020 – 2026.
   - **Hội nghị & bản sơ thảo:** Chỉ nhận bài hội nghị hàng đầu có phản biện kín (MICCAI, MLHC, NeurIPS, IEEE EMBC). Preprint (arXiv/medRxiv), editorial, letter và abstract hội nghị không có toàn văn bị loại.
   - **Nội dung:** Dự báo nguy cơ đột quỵ (incident stroke) từ dữ liệu bảng lâm sàng/EHR, có ít nhất một mô hình ML và báo cáo Recall/Sensitivity, AUC hoặc F1. Loại mô hình chỉ dùng ảnh (CT/MRI), tín hiệu (ECG/EEG) hoặc gen; loại dự báo kết cục sau đột quỵ.
   - **Toàn vẹn trích dẫn:** Mọi DOI được xác minh qua Crossref; bài bị rút (retraction) hoặc có expression of concern bị loại.
2. **Chiến lược tìm kiếm & PRISMA 2020 (truy cập ngày 2026-10-13):**
   - Europe PMC (`TITLE_ABS` stroke AND ML/RF/GB/XGBoost/stacking/ensemble AND prediction/risk AND imbalance/SMOTE/EHR/risk factors/Kaggle, 2020–2026, journal article): 499 bản ghi.
   - OpenAlex (3 truy vấn, lọc 2020–2026, có DOI, loại `article`): 300 bản ghi — bổ sung khi Semantic Scholar trả về HTTP 429.
   - Semantic Scholar (chạy lại cùng ngày): 1/4 truy vấn thành công, 100 bản ghi (74 bản ghi mới sau khử trùng lặp, không bài nào đạt tiêu chí); 3 truy vấn còn lại vẫn HTTP 429.
   - PubMed: E-utilities bị NCBI WAF chặn từ môi trường chạy; Europe PMC đã bao phủ phần lớn MEDLINE.
   - Nguồn khác (bài seed từ đề bài): 3 bản ghi.
   - Sau khử trùng lặp 2 tầng (DOI/PMID + RapidFuzz ≥ 92%): **808** bản ghi → loại **735** (kết cục sau đột quỵ, ảnh/tín hiệu, ngoài chủ đề, nơi xuất bản không đạt tiêu chí, abstract hội nghị) → **73** bài đủ điều kiện → **18** bài lõi được trích xuất 8 trường.
   - Crossref phát hiện và loại 3 bài bị rút cùng 1 bài có expression of concern: Hassan et al. (Sci Rep; Retraction Note 2026, cũng dùng dữ liệu Kaggle), Saleem et al. (Sci Rep 2024), Mochurad et al. (BMC Med Inform Decis Mak 2025; rút ngày 2026-06-15), Sahriar et al. (Heliyon 2024, expression of concern); thêm 1 bài XAI trên dữ liệu đột quỵ (Sci Rep 2025) đã bị rút, phát hiện ở lượt Semantic Scholar.
3. **8 trường thông tin bắt buộc trong Ma trận tổng quan y văn:**
   - (1) Tác giả & Năm xuất bản; (2) Tập bệnh nhân mục tiêu; (3) Nguồn dữ liệu & Kích thước mẫu; (4) Thuộc tính/Đặc trưng chính; (5) Kiến trúc mô hình ML/DL; (6) Giao thức phân chia dữ liệu (Data Split Protocol); (7) Độ đo & Kết quả chính (Recall, AUC, F1); (8) Hạn chế & Khoảng trống nghiên cứu.
4. **Nguyên tắc minh bạch trích dẫn (Traceability):**
   - Mọi số liệu trong ma trận khớp với trường `ground_truth_quote` (trích nguyên văn) trong `evidence_matrix.json`. Cột "Vị trí" ghi mục/đoạn/bảng trong toàn văn Europe PMC (bản PMC không có số trang); bài không lấy được toàn văn mở được trích ở mức Abstract và ghi rõ như vậy (cột "Vị trí"); với bài có PDF trong SSoT, vị trí ghi theo số trang PDF.

#### Bảng tổng quan Ma trận Y văn (Literature Matrix):

**Nhóm A — Bộ dữ liệu Kaggle, mất cân bằng lớp và ensemble**

| Tác giả & Năm | Tập bệnh nhân | Nguồn dữ liệu & Kích thước mẫu | Đặc trưng chính | Mô hình ML/DL | Giao thức chia dữ liệu | Độ đo & Kết quả | Hạn chế & Khoảng trống | Vị trí |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dritsas & Trigka (2022) [@Dritsas2022Stacking] | Người >18 tuổi | Kaggle, 3254 người; 6148 mẫu sau SMOTE | Tuổi, BMI, glucose, tăng huyết áp, bệnh tim, hút thuốc | Stacking (NB, RF, J48, RepTree; meta LR) | SMOTE toàn bộ dữ liệu → 10-fold CV | AUC 98.9%, Recall 97.4%, Accuracy 98% | SMOTE trước CV → rò rỉ dữ liệu, chỉ số bị thổi phồng; không kiểm định ngoài | PDF tr. 1 (Abstract), tr. 3 (Mục 3.1), tr. 8 (Mục 4.1), tr. 10 (Mục 5) |
| Rehman et al. (2023) [@Rehman2023RDET] | Kaggle | Kaggle, 3254 người (bộ gốc: 4861 bình thường, 249 đột quỵ) | 10 đặc trưng Kaggle | RDET stacking/soft voting + SMOTE/ADASYN | Oversample rồi mới chia 80/20 train/test | SMOTE: accuracy 100%, recall 100%; ADASYN: accuracy 96.4%, recall 97%; AUC 0.98 (RF, ETC) | Rò rỉ rõ ràng: dữ liệu được oversample trước khi chia nên bản sao tổng hợp của ca test nằm trong tập train | Abstract; Methods; ROC curves ¶3; Bảng 6 |
| Melnykova et al. (2025) [@Melnykova2025Imbalanced] | Kaggle | 4981 dòng, 11 cột | 10 đặc trưng Kaggle | RF (tốt nhất) + SMOTE, grid search | Chia 80-20 | Precision/Recall/F1 90%; RF không cân bằng Accuracy 96% | Accuracy 96% gây hiểu lầm trên dữ liệu lệch; thứ tự SMOTE so với chia dữ liệu không nêu rõ | Abstract; Methods; Bảng 4 |
| Kokkotis et al. (2022) [@Kokkotis2022Explainable] | Kaggle Cerebral Stroke | 43,400 người | 10 yếu tố nguy cơ | RF, XGBoost, KNN, SVM, MLP + random under-sampling; SHAP | Nested stratified 10-fold CV, tune trên fold train | LR: AUC 83.30, Sensitivity 78.12; MLP: Sensitivity 81.40, FN rate 18.60; XGBoost: AUC 82.50 | LR ngang/hơn mô hình phức tạp về AUC; under-sampling bỏ dữ liệu; không kiểm định ngoài | Data Description; Mục 2.6 ¶1; Bảng 3 |
| Akinwumi et al. (2025) [@Akinwumi2025Evaluating] | Kaggle | 5110 bản ghi | Tuổi, glucose, BMI | LR, RF, GB, SVM, KNN sau cân bằng lớp | 80/20 + 5-fold CV sau resampling | LR: Accuracy 0.9511 (95.11%), ROC-AUC 0.8362; tập test LR TP 0 / FN 50 | Recall rất kém: LR không phát hiện ca đột quỵ nào trong 50 ca test dù accuracy cao | PDF tr. 1 (Abstract), tr. 9, tr. 11, tr. 13 (Bảng 2) |

**Nhóm B — Cohort và EHR quy mô lớn**

| Tác giả & Năm | Tập bệnh nhân | Nguồn dữ liệu & Kích thước mẫu | Đặc trưng chính | Mô hình ML/DL | Giao thức chia dữ liệu | Độ đo & Kết quả | Hạn chế & Khoảng trống | Vị trí |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Chun et al. (2021) [@Chun2021CKB] | Người trưởng thành Trung Quốc | China Kadoorie Biobank ~0.5 triệu; train 174 498 nam, 253 766 nữ | Yếu tố nguy cơ tim mạch, lối sống | GBT; ensemble GBT + Cox | Train 85% / validation 12.75% / test 2.25%, mô hình theo giới | AUROC 0.833 (nam), 0.836 (nữ); Sensitivity 80% (nam), 74% (nữ) | Một quốc gia; lợi ích ensemble nhỏ | PDF tr. 1 (Abstract), tr. 2 (Methods), tr. 5 (Results) |
| Yang et al. (2021) [@Yang2021Hypertensive] | Bệnh nhân tăng huyết áp | EHR Thâm Quyến, 57,671 người (9421 ca đột quỵ) | Đặc trưng nền + xu hướng đa thời điểm | XGBoost, RF, SVM | Cân bằng ca-chứng → 19,953 mẫu, chia 7:3; 5-fold CV | XGBoost AUC 0.9220 (92.2%), Recall 0.8512; RF AUC 0.8956 (89.6%); LR AUC 0.8544, Recall 0.7141 | Tập test cân bằng nhân tạo, không phản ánh tỷ lệ mắc thật | PDF tr. 1 (Abstract), tr. 7 (Bảng 3), tr. 9 |
| Duan et al. (2025) [@Duan2025CHERRY] | Người 18–75 tuổi | CHERRY (Yinzhou), 92,172 người, 436 ca | 13 biến: tăng huyết áp, tuổi, đái tháo đường, HA tâm thu… | DT, RF, XGBoost, BPNN | Chia ngẫu nhiên 7:3; ngưỡng Youden | RF: AUC validation 0.980, Recall 0.947; LR: AUC 0.779 | RF, XGBoost hiệu chuẩn kém hơn LR; AUC rất cao với tỷ lệ mắc thấp gợi ý đánh giá nội bộ lạc quan | PDF tr. 1 (Abstract), tr. 4 (Results) |
| Vu et al. (2024) [@Vu2024Suita] | Cư dân đô thị Nhật Bản | Suita study, 7389 người, 53 biến | Tuổi, HA tâm thu, eGFR, hội chứng chuyển hóa, glucose | RF, SVM, XGBoost, LightGBM; SHAP | Down-sampling → chia 80/20 | RF: AUC 0.71, Recall 0.70; LR: AUC 0.68; SVM: AUC 0.73 | Trên cohort thật AUC chỉ ~0.7; down-sampling trước khi chia làm lệch tỷ lệ tập test | Abstract; Mục 3 ¶4; Bảng 3 |
| Lolak et al. (2023) [@Lolak2023Explainable] | Bệnh nhân nguy cơ cao (Thái Lan) | Ramathibodi, 275,247 người, 9659 ca (3.5%) | Rung nhĩ, tăng huyết áp, thuốc kháng tiểu cầu, HDL, tuổi | XGBoost, EBM, BN, TAN | Chia 80:20 theo số hồ sơ bệnh nhân (mỗi người chỉ ở một tập); MICE | XGBoost: C-statistic 0.89, F1 0.80; EBM 0.87; TAN 0.83; LR 0.80, F1 0.73 | Một bệnh viện; không báo cáo recall trong phần chính | Methods; Bảng 2 |
| Lip et al. (2022) [@Lip2022Multimorbidity] | Bệnh nhân đa bệnh (Mỹ) | 3,435,224 người | Bệnh đồng mắc động, nhân khẩu học | Gradient boosting / NN / LR | Train + external validation; decision curve | ML: c index 0.866; CHADS2 0.812; CHA2DS2-VASc 0.809 | Các cách tiếp cận ML không khác biệt có ý nghĩa | Abstract |

**Nhóm C — So sánh với thang điểm cổ điển, công bằng và diễn giải**

| Tác giả & Năm | Tập bệnh nhân | Nguồn dữ liệu & Kích thước mẫu | Đặc trưng chính | Mô hình ML/DL | Giao thức chia dữ liệu | Độ đo & Kết quả | Hạn chế & Khoảng trống | Vị trí |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Orfanoudaki et al. (2020) [@Orfanoudaki2020NonLinear] | Người >40 tuổi chưa đột quỵ | Framingham Offspring, 4,385 người (18,793 lượt khám); kiểm định 17,527 người | Biến của Revised FSRS, ECG, hematocrit | Optimal Classification Trees | Kiểm định ngoài trên cohort đa sắc tộc | C-statistic train 87.43% → validation 75.29%; R-FSRS 65.93% | Nhiều lượt khám/người; hiệu năng giảm mạnh khi kiểm định ngoài | Abstract; Derivation cohort ¶2; Validation cohort ¶1 |
| Hong et al. (2023) [@Hong2023Race] | Người da đen và da trắng ≥45 tuổi | Framingham, ARIC, MESA, REGARDS; 62,482 người | Yếu tố nguy cơ tự báo cáo | 2 thuật toán ML mới | Nguy cơ 10 năm; đánh giá theo chủng tộc, giới, tuổi | C index: Framingham 0.72, REGARDS 0.73, PCE 0.72 | ML không cải thiện phân biệt; mọi mô hình kém hơn ở người da đen | Abstract |
| Gao et al. (2024) [@Gao2024Fair] | Bệnh nhân rung nhĩ | All of Us (EHR) | Biến của CHADS2/CHA2DS2-VASc | LightGBM + giảm chênh lệch (ngưỡng theo nhóm) | Hold-out test theo thời gian, tăng tỷ lệ người da đen | AUROC từ gần 0.70 lên trên 0.80 | Chỉ quần thể rung nhĩ | Abstract |
| Wallace et al. (2023) [@Wallace2023VIMP] | Sleep Heart Health Study | Tổng quan 50 bài RF + minh họa | Chức năng phổi, thuốc, yếu tố nguy cơ | RF với knockoff VIMP | Không áp dụng | Gần như mọi bài trong 50 bài dùng VIMP có hạn chế đáng kể | OOB importance thiên lệch về biến tương quan | Abstract |

**Nhóm D — Tổng quan hệ thống/ tổng quan**

| Tác giả & Năm | Phạm vi | Nguồn & Số nghiên cứu | Trọng tâm | Mô hình | Phương pháp tổng hợp | Kết quả chính | Khoảng trống được nêu | Vị trí |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Soladoye et al. (2025) [@Soladoye2025SR] | ML dự báo đột quỵ 2013–2024 | 5 CSDL, 58 nghiên cứu (48 dùng EHR) | Thuật toán, dữ liệu, khu vực | Ensemble, DL | PRISMA, tổng hợp mô tả | Ensemble đạt accuracy 90.4–97.8% | Quần thể châu Phi xuất hiện trong ít hơn 4 nghiên cứu; báo cáo accuracy thay vì recall; các nghiên cứu hiệu năng cao thường dựa vào dữ liệu Kaggle không rõ nguồn gốc | PDF tr. 1 (Abstract), tr. 8 (Discussion) |
| Heseltine-Carpenter et al. (2025) [@HeseltineCarp2025SR] | ML trên dữ liệu bệnh viện thường quy | PubMed, 49 nghiên cứu | So với CHA2DS2-VASc, QRISK3 | Nhiều loại ML | Tổng quan có thẩm định | AUC từ 0.64 đến 0.99 | Nghi ngờ overfitting, thiếu hiệu chuẩn, kiểm định ngoài và giải thích | Abstract |
| Velagapudi et al. (2021) [@Velagapudi2021Discrepancies] | Nghiên cứu ML đột quỵ tại Mỹ | PubMed, 27 nghiên cứu | Nguồn gốc dữ liệu | Không áp dụng | Đối chiếu với tỷ lệ đột quỵ của CDC | 25.9% dùng dữ liệu California | Dữ liệu huấn luyện không khớp nơi gánh nặng đột quỵ cao nhất | Abstract |

**Bảng so sánh: xử lý mất cân bằng và rủi ro rò rỉ dữ liệu (14 nghiên cứu gốc; "—" = không báo cáo trong phần đã trích xuất)**

| Tác giả & Năm | Quy mô mẫu (N) | Ca dương tính / tỷ lệ | Thuật toán tốt nhất | Recall / Sensitivity | ROC-AUC / C-statistic | Xử lý mất cân bằng | Rủi ro rò rỉ dữ liệu |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dritsas & Trigka (2022) [@Dritsas2022Stacking] | 3254 | 5.53% | Stacking | 97.4% | 98.9% | SMOTE trên toàn bộ dữ liệu (6148 mẫu) | **Cao** — SMOTE trước 10-fold CV |
| Rehman et al. (2023) [@Rehman2023RDET] | 3254 | 249 / 4861 (bộ Kaggle gốc) | RDET stacking | 100% (SMOTE); 97% (ADASYN) | 0.98 (RF, ETC) | SMOTE, ADASYN | **Cao** — oversample rồi mới chia 80/20 |
| Melnykova et al. (2025) [@Melnykova2025Imbalanced] | 4981 | — | RF | 90% | — | SMOTE | **Chưa xác định** — thứ tự SMOTE/chia không rõ |
| Kokkotis et al. (2022) [@Kokkotis2022Explainable] | 43,400 | — | MLP (FN rate 18.60) / LR (AUC 83.30) | 81.40 (MLP) | 83.30 (LR) | Random under-sampling mỗi lần chạy | **Thấp** — nested stratified 10-fold CV |
| Akinwumi et al. (2025) [@Akinwumi2025Evaluating] | 5110 | 249 (4.9%); 50 ca trong tập test | LR (ROC-AUC 0.8362) | LR: 0 / 50 ca | 0.8362 | Resampling (không dùng SMOTE) | **Trung bình** — CV sau resampling |
| Chun et al. (2021) [@Chun2021CKB] | ~0.5 triệu (train 174 498 nam, 253 766 nữ) | — | GBT / ensemble GBT + Cox | 80% (nam), 74% (nữ) | 0.833 (nam), 0.836 (nữ) | Không resampling | **Thấp** — train/validation/test tách biệt |
| Yang et al. (2021) [@Yang2021Hypertensive] | 57,671 | 9421 ca | XGBoost | 0.8512 | 0.9220 | Lấy mẫu cân bằng ca-chứng → 19,953 mẫu | **Trung bình** — tập test cân bằng nhân tạo |
| Duan et al. (2025) [@Duan2025CHERRY] | 92,172 | 436 ca | RF | 0.947 | 0.980 | — | **Trung bình** — kết quả nội bộ rất lạc quan, RF hiệu chuẩn kém |
| Vu et al. (2024) [@Vu2024Suita] | 7389 | — | RF | 0.70 | 0.71 | Down-sampling | **Trung bình** — down-sampling trước khi chia |
| Lolak et al. (2023) [@Lolak2023Explainable] | 275,247 | 9659 ca (3.5%) | XGBoost (F1 0.80) | — | 0.89 | — | **Thấp** — chia 80:20 theo bệnh nhân |
| Lip et al. (2022) [@Lip2022Multimorbidity] | 3,435,224 | — | ML (LR/GB/NN) | — | 0.866 | — | **Thấp** — external validation |
| Orfanoudaki et al. (2020) [@Orfanoudaki2020NonLinear] | 4,385 (18,793 lượt khám) | — | Optimal Classification Trees | — | 75.29% (kiểm định ngoài) | — | **Thấp** — cohort kiểm định riêng |
| Hong et al. (2023) [@Hong2023Race] | 62,482 | — | Không mô hình nào vượt trội | — | 0.72–0.73 | — | **Thấp** — 4 cohort, đánh giá theo nhóm |
| Gao et al. (2024) [@Gao2024Fair] | — | — | LightGBM + giảm chênh lệch | — | trên 0.80 | — | **Thấp** — hold-out theo thời gian |
| **Dự án N22 (tham chiếu)** | 5,110 (test 1,022) | 249 (4.9%) | Stacking (OOF theo bệnh nhân) | 0.88 trên tập test | 0.83 | SMOTENC trong pipeline | **Thấp** — chia theo bệnh nhân, chỉ fit trên train |

*PR-AUC không được báo cáo ở các nghiên cứu trên trong phần đã trích xuất; dự án N22 báo cáo AP = 0.17 (Stacking) trên tập test.*

### Câu 2.2: Phân tích khoảng trống nghiên cứu & Mô hình Baseline
1. **Khoảng trống nghiên cứu chính (Research Gaps)** — đã qua Chốt 3.5 (đối chiếu cả bằng chứng phản bác):
   - **Rò rỉ dữ liệu do cân bằng lớp trước khi chia dữ liệu.** Dritsas & Trigka chạy SMOTE trên toàn bộ dữ liệu rồi mới 10-fold CV trên 6148 mẫu và báo cáo AUC 98.9% [@Dritsas2022Stacking]; Rehman et al. oversample rồi mới chia train/test và báo cáo accuracy 100% [@Rehman2023RDET]; Vu et al. down-sampling trước khi chia [@Vu2024Suita]; Yang et al. đánh giá trên tập test đã cân bằng nhân tạo [@Yang2021Hypertensive]. Ngược lại, Kokkotis et al. dùng nested stratified 10-fold CV và chỉ đạt AUC 83.30 [@Kokkotis2022Explainable]. Khi dữ liệu có nhiều bản ghi cho một người (Orfanoudaki et al.: 4,385 người, 18,793 lượt khám [@Orfanoudaki2020NonLinear]), chia theo bản ghi còn làm rò rỉ thông tin bệnh nhân; Lolak et al. là ví dụ chia 80:20 theo bệnh nhân trên 275,247 người [@Lolak2023Explainable]. Bộ Kaggle có một dòng cho mỗi `id`, nên chia theo bệnh nhân ở dự án này là biện pháp phòng ngừa có nguyên tắc, còn nguồn rò rỉ chính cần chặn là resampling ngoài pipeline.
   - **Đánh đổi Accuracy và Recall.** Akinwumi et al. đạt accuracy 0.9511 nhưng LR không phát hiện ca đột quỵ nào (TP 0, FN 50) trên tập test [@Akinwumi2025Evaluating]; Melnykova et al. cho thấy accuracy 96% gây hiểu lầm trên dữ liệu lệch [@Melnykova2025Imbalanced]; tổng quan của Soladoye et al. vẫn tổng hợp hiệu năng chủ yếu bằng accuracy (90.4–97.8%) [@Soladoye2025SR].
   - **Ưu thế của ML co lại khi đánh giá nghiêm ngặt.** Trên 62,482 người, ML không cải thiện phân biệt so với pooled cohort equations (C index 0.72) [@Hong2023Race]; LR có AUC 83.30, cao nhất trong nghiên cứu của Kokkotis et al. [@Kokkotis2022Explainable]; trên cohort Suita, RF chỉ đạt AUC 0.71 so với 0.68 của LR [@Vu2024Suita]; mô hình cây của Orfanoudaki et al. giảm từ 87.43% xuống 75.29% khi kiểm định ngoài [@Orfanoudaki2020NonLinear]. Các nghiên cứu cho thấy ML vượt trội rõ rệt thường là trên cohort rất lớn với đánh giá cẩn thận, như GBT với AUROC 0.833 (nam) và 0.836 (nữ) [@Chun2021CKB], hoặc c index 0.866 so với 0.812 của CHADS2 [@Lip2022Multimorbidity].
   - **Giải thích, hiệu chuẩn và công bằng — không chỉ "có SHAP".** SHAP đã phổ biến [@Kokkotis2022Explainable; @Vu2024Suita], nhưng gần như mọi bài trong 50 bài RF được tổng quan có hạn chế đáng kể khi dùng độ quan trọng biến [@Wallace2023VIMP]. RF và XGBoost hiệu chuẩn kém hơn LR [@Duan2025CHERRY]; tổng quan 49 nghiên cứu nêu thiếu hiệu chuẩn, kiểm định ngoài và giải thích [@HeseltineCarp2025SR]. Mọi thuật toán phân biệt kém hơn ở người da đen [@Hong2023Race]; Gao et al. chỉ ra có thể giảm chênh lệch bằng ngưỡng theo nhóm, với AUROC trên 0.80 [@Gao2024Fair]; dữ liệu huấn luyện thường không đại diện cho nơi gánh nặng đột quỵ cao (25.9% từ California) [@Velagapudi2021Discrepancies].
2. **Hệ quả cho thiết kế của dự án N22:** SMOTENC đặt trong pipeline và chỉ fit trên fold train; chia theo bệnh nhân bằng `StratifiedGroupKFold`; ngưỡng quyết định chọn trên validation theo F2/recall; báo cáo Recall, AP (PR-AUC), F2 và ROC-AUC thay vì accuracy; LR luôn là baseline bắt buộc; bổ sung hiệu chuẩn và đánh giá theo nhóm (giới, tuổi) là hướng mở rộng.
3. **Mô hình cơ sở (Baseline Models):**
   - **Logistic Regression (LR):** baseline tuyến tính bắt buộc, vì trong nhiều nghiên cứu LR ngang các mô hình phức tạp [@Kokkotis2022Explainable; @Hong2023Race].
   - **Framingham Stroke Risk Profile (FSRP) / Cox Proportional Hazards:** thang điểm cổ điển; FSRP sửa đổi đạt C-statistic 65.93% khi kiểm định ngoài [@Orfanoudaki2020NonLinear], mô hình Framingham đạt C index 0.72 [@Hong2023Race]; Cox là đối chứng của GBT trong CKB [@Chun2021CKB]. Bộ dữ liệu Kaggle thiếu huyết áp tâm thu và cholesterol nên không tính trực tiếp được FSRP; LR trên cùng tập biến là baseline thay thế.

### Câu 2.3: Quản lý trích dẫn & Thư mục SSoT
1. **Thư mục Single Source of Truth (SSoT):**
   - PDF gốc lưu tại `data/SSoT_pdf_references/` (không commit vì bản quyền; xem `README.md` trong thư mục). Metadata và trích dẫn nằm trong `docs/literature/evisyn/references.bib` (18 mục, sinh tự động từ Crossref theo DOI) và `evidence_matrix.json`.
   - Quy ước đặt tên file: `[Năm]_[HọTácGiả]_[TênBàiBáo_ViếtTắt].pdf` (ví dụ: `2022_Dritsas_StrokeRiskPredictionML.pdf`).
2. **Kiểm toán ngược:** Phần II được kiểm bằng `WritingPapers/scripts/verify_evidence_integrity.py` (mọi citekey phải có trong `references.bib`, mọi số liệu trong đoạn có trích dẫn phải khớp `evidence_matrix.json`).
3. **Chế tài xử phạt vi phạm:**
   - Trừ điểm nặng hoặc hủy bỏ kết quả đối với các hành vi: bịa đặt nguồn trích dẫn, trích dẫn không có file PDF lưu trữ trong thư mục SSoT, hoặc số liệu trong báo cáo sai lệch so với văn bản gốc.

---

## PHẦN III: EXPLORATORY DATA ANALYSIS - EDA (3.0 điểm)

### Câu 3.1: Đặc tả thuộc tính dữ liệu & Thiên vị dịch tễ
1. **Đặc tả bộ dữ liệu Kaggle Stroke Prediction Dataset:**
   - **Tổng số mẫu:** 5,110 bản ghi bệnh nhân.
   - **Số thuộc tính:** 12 (10 đặc trưng đầu vào, 1 định danh `id`, 1 nhãn mục tiêu `stroke`).
2. **Phân tích thiên vị nhân khẩu học & Yếu tố nguy cơ:**
   - **Tuổi (`age`):** Là yếu tố có tương quan cao nhất với biến cố đột quỵ. Nhóm tuổi từ 60–80 có mật độ mắc đột quỵ cao vượt trội. Bệnh nhân dưới 40 tuổi có tỷ lệ đột quỵ gần như bằng 0.
   - **Giới tính (`gender`):** Tỷ lệ phân bố Nam/Nữ tương đối cân bằng (Nữ ~58.6%, Nam ~41.4%), không cho thấy sự thiên vị lệch lạc nghiêm trọng theo giới.
   - **Mức đường huyết (`avg_glucose_level`):** Bệnh nhân có chỉ số đường huyết trung bình >200 mg/dL có nguy cơ đột quỵ cao gấp 3 lần so với nhóm có chỉ số đường huyết bình thường (<100 mg/dL).
   - **Tiền sử bệnh lý (`hypertension`, `heart_disease`):** Nhóm có tiền sử cao huyết áp hoặc bệnh tim có tỷ lệ bị đột quỵ cao hơn rõ rệt (chiếm >20% trong tổng số ca đột quỵ).

### Câu 3.2: Thách thức dữ liệu y tế
1. **Mất cân bằng lớp nghiêm trọng (Class Imbalance):**
   - Số ca không đột quỵ ($Y=0$): **4,861 mẫu (95.1%)**.
   - Số ca có đột quỵ ($Y=1$): **249 mẫu (4.9%)**.
   - **Phân biệt Sample-level vs Patient-level Imbalance:** Sample-level imbalance là sự lệch nhãn giữa các bản ghi; Patient-level imbalance là sự lệch nhãn giữa các bệnh nhân duy nhất (đặc biệt quan trọng khi một bệnh nhân có nhiều bản ghi theo thời gian).
2. **Nhiễu và dữ liệu khuyết thiếu (Data Quality Issues):**
   - **Khuyết thiếu chỉ số BMI (`bmi`):** 201 bản ghi thiếu giá trị BMI (~3.9% dữ liệu).
   - **Dữ liệu không rõ ràng (`Unknown` in `smoking_status`):** 1,544 bản ghi mang giá trị `Unknown` (~30.2% dữ liệu), đại diện cho thách thức lớn trong việc mã hóa thông tin.
3. **Chế tài mã nguồn EDA:**
   - Mã nguồn EDA bắt buộc phải đóng gói thành các hàm có khả năng tái sử dụng, thực thi end-to-end không phát sinh lỗi.

---

## PHẦN IV: ADVANCED PREPROCESSING & ZERO DATA LEAKAGE (+2.0 điểm)

### Câu 4.1: Quy trình tiền xử lý dữ liệu bảng/EHR
1. **Nội suy dữ liệu khuyết thiếu (Missing Value Imputation):**
   - Không xóa bỏ 201 mẫu thiếu BMI để tránh mất mát dữ liệu lớp hiếm ($Y=1$).
   - Áp dụng **KNN Imputer** (dựa trên các bệnh nhân tương đồng về `age`, `hypertension`, `avg_glucose_level`) hoặc **Median Imputation** tính riêng theo nhóm tuổi.
2. **Mã hóa đặc trưng phân loại (Categorical Encoding):**
   - **One-Hot Encoding:** Áp dụng cho các biến danh nghĩa không có thứ tự: `gender`, `work_type`, `Residence_type`, `smoking_status`.
   - **Binary Encoding:** Áp dụng cho `ever_married` (`Yes`: 1, `No`: 0).
3. **Chuẩn hóa đặc trưng liên tục (Feature Scaling):**
   - Áp dụng **StandardScaler (Z-score Normalization)** cho `age`, `avg_glucose_level`, `bmi` để đưa về mean = 0, std = 1.
4. **Cân bằng lớp bằng SMOTENC trước One-Hot Encoding:**
   - Impute trên Train trước SMOTENC; khai báo các biến categorical và binary là categorical để không sinh giá trị phân số. Stacking tạo meta-feature OOF bằng StratifiedGroupKFold theo Patient ID, fit toàn bộ pipeline trong từng fold.
   - Sinh mẫu nhân tạo cho lớp thiểu số ($Y=1$) trên tập Train để đưa tỷ lệ về mức cân bằng (50:50 hoặc 30:70), giúp mô hình học được biên phân chia hiệu quả.

### Câu 4.2: Kỷ luật chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage)
1. **Tác hại của việc phân chia ngẫu nhiên cấp độ mẫu/bản ghi:**
   - Nếu một bệnh nhân có nhiều bản ghi mà dữ liệu bị chia ngẫu nhiên vào cả tập Train và Test, mô hình sẽ "học thuộc lòng" đặc trưng riêng của bệnh nhân đó, dẫn đến điểm đánh giá tập Test cao giả tạo nhưng thất bại hoàn toàn khi triển khai thực tế.
2. **Phương pháp phân chia độc lập (Patient-level Splitting):**
   - Sử dụng **StratifiedGroupKFold** hoặc **GroupShuffleSplit** với `groups = patient_id` và `stratify = stroke`.
   - Đảm bảo toàn bộ các bản ghi của cùng một `patient_id` chỉ nằm duy nhất trong tập Train, Validation, hoặc Test.
3. **Quy tắc Fit-on-Train-Only:**
   - Các thao tác: `Imputer.fit()`, `Scaler.fit()`, `Encoder.fit()`, và `SMOTE.fit_resample()` **CHỈ ĐƯỢC PHÉP THỰC HIỆN TRÊN TẬP TRAIN**.
   - Tập Validation và Test chỉ sử dụng hàm `.transform()` từ các bộ đã fit trên tập Train.

4. **Đoạn mã Python Assertion Code kiểm chứng Zero Data Leakage:**

```python
from src.data_loader import split_features_target
from src.models import get_model
from src.preprocessing import (
    make_full_pipeline, patient_level_split, verify_zero_data_leakage,
)


def run_clean_preprocessing_pipeline(df):
    split = patient_level_split(df)
    verify_zero_data_leakage(split.train, split.val, split.test)
    X_train, y_train = split_features_target(split.train)
    X_test, _ = split_features_target(split.test)
    # Impute -> SMOTENC -> One-Hot/Scaler -> model; fit chỉ trên Train.
    pipeline = make_full_pipeline(get_model("logistic_regression"))
    pipeline.fit(X_train, y_train)
    return pipeline, pipeline.predict_proba(X_test)
```

---

## KIẾN TRÚC PROJECT VÀ CẤU TRÚC REPOSITORY GITHUB

Dưới đây là sơ đồ cấu trúc thư mục quy chuẩn để khởi tạo dự án trên GitHub và kết hợp làm việc song song với Claude / Codex / Google Antigravity:

```
stroke-prediction-n22/
│
├── .github/
│   └── workflows/
│       └── ci_cd_pipeline.yml       # Tự động hóa kiểm thử mã nguồn & Linting
│
├── data/
│   ├── raw/                         # Dữ liệu gốc Kaggle (healthcare-dataset-stroke-data.csv)
│   ├── processed/                   # Dữ liệu đã qua tiền xử lý
│   └── SSoT_pdf_references/         # Lưu trữ PDF các bài báo y văn chuẩn hóa
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_imbalance.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Nạp dữ liệu và kiểm tra cấu trúc
│   ├── preprocessing.py             # Imputer, Scaler, Encoder chuẩn Zero Data Leakage
│   ├── models.py                    # Khởi tạo Logistic Regression, Random Forest, XGBoost
│   ├── evaluation.py                # Tính Recall, PR-AUC, Confusion Matrix, Assertion Leakage
│   └── explainability.py            # SHAP / LIME cho tính giải thích lâm sàng (XAI)
│
├── tests/
│   ├── test_data_leakage.py         # Unit test kiểm tra isdisjoint Patient IDs
│   └── test_preprocessing.py        # Unit test cho Imputer và Scaler
│
├── docs/
│   └── DETAILED_EXAM_REPORT.md      # Báo cáo lời giải chi tiết (Nội dung file này)
│
├── requirements.txt                 # Danh sách các thư viện Python (scikit-learn, imblearn, shap...)
├── .gitignore
└── README.md                        # Hướng dẫn chạy dự án & Tóm tắt kết quả
```
