# Paper Summary: Kokkotis et al. (Diagnostics 2022)

**Tiêu đề**: An Explainable Machine Learning Pipeline for Stroke Prediction on Imbalanced Data  
**Tác giả**: Kokkotis, Christos; Giarmatzis, Georgios; Giannakou, Erasmia; Moustakidis, Serafeim; Tsatalas, Themistoklis; Tsiptsios, Dimitrios; Vadikolias, Konstantinos; Aggelousis, Nikolaos  
**Tạp chí**: *Diagnostics* (2022; 12(10): 2392) — DOI: https://doi.org/10.3390/diagnostics12102392  
**Nhóm phân loại**: Dữ liệu mở Kaggle  
**Citekey**: `Kokkotis2022Explainable` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC9600473: Data Description; Sec 2.6 para 1; Sec 4 para 2; Table 3

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 43,400 người, 10 yếu tố nguy cơ (Kaggle “Cerebral Stroke Prediction-Imbalanced Dataset”).
- Số ca dương tính / tỷ lệ: Không báo cáo trong phần đã trích xuất.
- Thiết kế: benchmark trên dữ liệu công khai.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 10 yếu tố nguy cơ: tuổi, BMI, giới, glucose trung bình, loại công việc, tăng huyết áp, bệnh tim, hút thuốc…
- Xử lý dữ liệu thiếu: Không báo cáo trong phần đã trích xuất.
- **Mất cân bằng:** random under-sampling trong mỗi lần chạy.
- **Chia dữ liệu & rò rỉ:** nested stratified 10-fold CV; GridSearchCV chỉ fit trên fold huấn luyện (Mục 2.6 ¶1) → rủi ro rò rỉ **thấp**.

## 3. Thuật toán & Kết quả thực nghiệm
- LR, RF, XGBoost, KNN, SVM, MLP.
- Bảng 3: LR AUC 83.30, Sensitivity 78.12; XGBoost AUC 82.50; MLP Sensitivity 81.40, FN rate 18.60.
- Mô hình tốt nhất theo FN rate: MLP; theo AUC: LR.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: SHAP.
- Điểm mạnh: quy trình đánh giá đúng (nested CV), báo cáo sensitivity và FN rate thay vì chỉ accuracy.
- Hạn chế: LR ngang/hơn mô hình phức tạp; under-sampling bỏ dữ liệu; không kiểm định ngoài. Bài học cho N22: LR là baseline bắt buộc.

## Trích dẫn nguyên văn (Ground truth)
> "The dataset contains a total of 43,400 subjects and 10 risk factors" (Data Description); "a nested stratified 10-fold cross-validation process was adopted" (Section 2.6 para 1); Table 3: LR AUC 83.30, sensitivity 78.12; MLP sensitivity 81.40, FN rate 18.60
