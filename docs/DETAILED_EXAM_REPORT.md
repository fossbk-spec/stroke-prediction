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

### Câu 2.1: Ma trận y văn & Tiêu chuẩn thẩm định
1. **Tiêu chuẩn lựa chọn bài báo khoa học:**
   - **Xếp hạng:** Thuộc danh mục SJR Q1/Q2 (Scopus / Web of Science).
   - **Thời gian xuất bản:** Trong giai đoạn 2020 – 2026.
   - **Quy định với Hội nghị & Bản sơ thảo:** Các bài báo tại hội nghị hàng đầu (MICCAI, MLHC, NeurIPS, IEEE EMBC) bắt buộc phải qua quy trình phản biện kín độc lập (Double-blind Peer Review). Các bản sơ thảo preprint trên arXiv chỉ đóng vai trò tham khảo bổ trợ, không được xem là nguồn chứng cứ khoa học chính thức.
2. **8 trường thông tin bắt buộc trong Ma trận tổng quan y văn:**
   - (1) Tác giả & Năm xuất bản; (2) Tập bệnh nhân mục tiêu; (3) Nguồn dữ liệu & Kích thước mẫu; (4) Thuộc tính/Đặc trưng chính; (5) Kiến trúc mô hình ML/DL; (6) Giao thức phân chia dữ liệu (Data Split Protocol); (7) Độ đo & Kết quả chính (Recall, AUC, F1); (8) Hạn chế & Khoảng trống nghiên cứu.
3. **Nguyên tắc minh bạch trích dẫn (Traceability):**
   - Mọi số liệu thực nghiệm (AUC, Recall, Kích thước mẫu) trích dẫn trong ma trận phải ghi rõ vị trí xuất hiện trong văn bản gốc (ví dụ: Trang 4, Bảng 2, Dòng 10).

#### Bảng tổng quan Ma trận Y văn (Literature Matrix):
| Tác giả & Năm | Tập bệnh nhân | Nguồn dữ liệu & Kích thước mẫu | Đặc trưng chính | Mô hình ML/DL | Giao thức chia dữ liệu | Độ đo & Kết quả | Hạn chế & Khoảng trống |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Garg et al. (2025)** | Quần thể lâm sàng chung | Kaggle Stroke Dataset (5,110 mẫu) | Age, BMI, Avg Glucose, Hypertension, Heart Disease | Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN | 80/20 Train/Test Random Split | ROC-AUC: 0.84, Accuracy: 95%, **Recall ca bệnh rất thấp (<0.20)** | Rò rỉ dữ liệu tiềm ẩn do chia ngẫu nhiên; Recall lớp dương kém do chưa xử lý imbalance triệt để. |
| **Abed et al. (2022)** | Bệnh nhân tầm soát nguy cơ | Dataset đa nguồn (5,110 mẫu) | Age, Hypertension, Heart Disease, Glucose, Smoking | Stacking Ensemble (Base: RF, XGBoost, SVM; Meta: Logistic Regression) | 10-Fold Cross-Validation kết hợp SMOTE | AUC: 98.9%, Accuracy: 98.0%, Recall: 97.5% | Áp dụng SMOTE trên toàn bộ dataset trước khi CV gây **Data Leakage** nghiêm trọng, thổi phồng AUC. |
| **Shao et al. (2021)** | Bệnh nhân cao huyết áp | Big Data Y tế Trung Quốc (>100,000 mẫu EHR) | Huyết áp đa thời gian, Tuổi, Đái tháo đường, Lipid máu | XGBoost, Gradient Boosted Trees, Survival Analysis | Prospective Cohort Split (Train/Test theo trung tâm/thời gian) | C-index: 0.82, Recall: 81.2% | Phụ thuộc dữ liệu chuỗi thời gian đắt đỏ; khó triển khai ở các phòng khám cơ sở thiếu EHR đầy đủ. |
| **N22 Team (2025)** | Quần thể Kaggle Stroke | Kaggle Stroke Prediction (5,110 mẫu) | Age, BMI, Glucose, Hypertension, Heart Disease | Random Forest, Decision Tree, Logistic Regression, SMOTE | Stratified Train/Test Split | Recall (SMOTE+RF): >85%, ROC-AUC: 0.83 | Chưa thử nghiệm Stacking Ensemble nâng cao; thiếu bước kiểm chứng độc lập trên tập Patient-level ID. |

### Câu 2.2: Phân tích khoảng trống nghiên cứu & Mô hình Baseline
1. **Khoảng trống nghiên cứu chính (Research Gaps):**
   - **Rò rỉ dữ liệu (Data Leakage) do phân chia cấp độ mẫu (Sample-level Split):** Nhiều nghiên cứu chia dữ liệu ngẫu nhiên trước khi áp dụng SMOTE hoặc chuẩn hóa, khiến thông tin tập Test bị rò rỉ vào tập Train.
   - **Thiếu tính giải thích lâm sàng (XAI - Explainable AI):** Các mô hình Ensemble phức tạp đạt AUC cao nhưng hoạt động như "hộp đen" (Black-box), gây e ngại cho bác sĩ lâm sàng khi đưa ra quyết định dự phòng.
   - **Sự đánh đổi giữa Accuracy và Recall:** Hầu hết các mô hình đạt Accuracy >95% nhưng có Recall cực kỳ thấp (<20%) đối với nhóm bệnh nhân thực sự bị đột quỵ.
2. **Mô hình cơ sở (Baseline Models):**
   - **Logistic Regression (LR):** Mô hình học máy tuyến tính tiêu chuẩn.
   - **Framingham Stroke Risk Profile (FSRP) / Mô hình Cox Proportional Hazards:** Thang điểm dự báo nguy cơ kinh điển trong y học dựa trên độ tuổi, huyết áp, tiểu đường và hút thuốc.

### Câu 2.3: Quản lý trích dẫn & Thư mục SSoT
1. **Thư mục Single Source of Truth (SSoT):**
   - Lưu trữ toàn bộ tệp PDF tài liệu gốc tại thư mục `/knowledge/pdf_references/`.
   - Quy ước đặt tên file chuẩn hóa: `[Năm]_[HọTácGiả]_[TênBàiBáo_ViếtTắt].pdf` (ví dụ: `2025_Garg_EvaluatingMLStrokePrediction.pdf`).
2. **Chế tài xử phạt vi phạm:**
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
