# Paper Summary: Rehman et al. (PeerJ Computer Science 2023)

**Tiêu đề**: RDET stacking classifier: a novel machine learning based approach for stroke prediction using imbalance data  
**Tác giả**: Rehman, Amjad; Alam, Teg; Mujahid, Muhammad; Alamri, Faten S.; Ghofaily, Bayan Al; Saba, Tanzila  
**Tạp chí**: *PeerJ Computer Science* (2023; 9: e1684) — DOI: https://doi.org/10.7717/peerj-cs.1684  
**Nhóm phân loại**: Dữ liệu mở Kaggle  
**Citekey**: `Rehman2023RDET` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC10703010: Abstract; ensemble evaluation para 2; ROC curves para 3; Table 6

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 3,254 người (bộ Kaggle sau khi lọc).
- Bộ Kaggle gốc mô tả trong bài: 4,861 người bình thường và 249 người đột quỵ.
- Thiết kế: benchmark trên dữ liệu công khai.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 10 đặc trưng Kaggle.
- Xử lý dữ liệu thiếu: Không báo cáo trong phần đã trích xuất.
- **Mất cân bằng:** SMOTE và ADASYN tạo bộ dữ liệu cân bằng.
- **Chia dữ liệu & rò rỉ:** "the oversampled data is divided into training and testing sets" (80/20) → rò rỉ **rõ ràng**: bản sao tổng hợp của ca test nằm trong tập train.

## 3. Thuật toán & Kết quả thực nghiệm
- Đơn lẻ: RF, ETC, AdaBoost, SGD, KNN, DT; Ensemble: RDET stacking và soft voting.
- Bảng 6: Stacking + SMOTE accuracy 100%, recall 100%; Stacking + ADASYN accuracy 96.4%, recall 97%; RF, ETC: AUC 0.98 với SMOTE (ROC curves ¶3).

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: không.
- Hạn chế: accuracy 100% với outcome ~5% là bất hợp lý, phù hợp với rò rỉ do oversampling trước khi đánh giá; không có tập test độc lập. Bài học cho N22: SMOTENC phải nằm trong pipeline.

## Trích dẫn nguyên văn (Ground truth)
> "the total number of participants is 3,254" (Abstract); "The stacking method achieved a remarkable accuracy of 100% as well as exceptional F1-score, precision, and recall score" (Results); "As with using SMOTE for balancing the dataset, RF, ETC achieved a 0.98 AUC score" (ROC curves para 3); "Next, the oversampled data is divided into training and testing sets" and "A total of 80% of the data is employed for training the classifiers and 20% for testing" (Methods); Table 6: SMOTE stacking accuracy 100, recall 100; ADASYN stacking accuracy 96.4, recall 97; "4,861 individuals classified as normal (healthy) and 249 individuals with a stroke" (Dataset)
