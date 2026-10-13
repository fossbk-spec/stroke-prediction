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
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: benchmark trên dữ liệu công khai.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 10 đặc trưng Kaggle.
- Xử lý dữ liệu thiếu: Không báo cáo trong phần đã trích xuất.
- **Mất cân bằng:** SMOTE và ADASYN tạo bộ dữ liệu cân bằng.
- **Chia dữ liệu & rò rỉ:** đánh giá 10-fold CV trên bộ dữ liệu đã oversample → rủi ro rò rỉ **cao**.

## 3. Thuật toán & Kết quả thực nghiệm
- Đơn lẻ: RF, ETC, AdaBoost, SGD, KNN, DT; Ensemble: RDET stacking và soft voting.
- Stacking: accuracy 100%; RF, ETC: AUC 0.98 với SMOTE (ROC curves ¶3).

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: không.
- Hạn chế: accuracy 100% với outcome ~5% là bất hợp lý, phù hợp với rò rỉ do oversampling trước khi đánh giá; không có tập test độc lập. Bài học cho N22: SMOTENC phải nằm trong pipeline.

## Trích dẫn nguyên văn (Ground truth)
> "the total number of participants is 3,254" (Abstract); "The stacking method achieved a remarkable accuracy of 100% as well as exceptional F1-score, precision, and recall score" (Results); "As with using SMOTE for balancing the dataset, RF, ETC achieved a 0.98 AUC score" (ROC curves para 3)
