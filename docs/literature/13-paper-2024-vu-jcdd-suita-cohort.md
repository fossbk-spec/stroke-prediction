# Paper Summary: Vu et al. (Journal of Cardiovascular Development and Disease 2024)

**Tiêu đề**: Machine Learning Approaches for Stroke Risk Prediction: Findings from the Suita Study  
**Tác giả**: Vu, Thien; Kokubo, Yoshihiro; Inoue, Mai; Yamamoto, Masaki; Mohsen, Attayeb; Martin-Morales, Agustin; Inoué, Takao; Dawadi, Research; Araki, Michihiro  
**Tạp chí**: *Journal of Cardiovascular Development and Disease* (2024; 11(7): 207) — DOI: https://doi.org/10.3390/jcdd11070207  
**Nhóm phân loại**: Dữ liệu cohort  
**Citekey**: `Vu2024Suita` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC11276746: Abstract; Sec 2.7 para 5; Sec 3 para 4; Table 3

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 7,389 người, 53 biến (Suita study, Nhật Bản).
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: cohort dân số tiến cứu.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 53 biến lâm sàng, xét nghiệm, nhân trắc.
- Xử lý dữ liệu thiếu: Không báo cáo trong phần đã trích xuất.
- **Mất cân bằng:** down-sampling nhóm không đột quỵ.
- **Chia dữ liệu & rò rỉ:** down-sampling rồi mới chia ngẫu nhiên 80/20 → tập test bị đổi tỷ lệ; rủi ro **trung bình**.

## 3. Thuật toán & Kết quả thực nghiệm
- LR, RF, SVM, XGBoost, LightGBM; k-prototype clustering.
- Bảng 3: RF AUC 0.71, Recall 0.70; LR AUC 0.68; SVM AUC 0.73.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: SHAP (tuổi, HA tâm thu, tăng huyết áp, eGFR, hội chứng chuyển hóa, glucose).
- Điểm mạnh: dữ liệu cohort thật.
- Hạn chế: AUC chỉ ~0.7 trên cohort thật, thấp hơn nhiều so với các bài Kaggle. Bài học: không kỳ vọng AUC >0.9 trên dữ liệu thực.

## Trích dẫn nguyên văn (Ground truth)
> "comprising 7389 participants and 53 variables" (Abstract); Table 3: RF AUC 0.71 ± 0.06, recall 0.70 ± 0.05; LR AUC 0.68 ± 0.06
