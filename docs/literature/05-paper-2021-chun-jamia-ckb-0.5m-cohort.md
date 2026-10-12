# Paper Summary: Chun et al. (JAMIA 2021)

**Tiêu đề**: Stroke risk prediction using machine learning: a prospective cohort study of 0.5 million Chinese adults  
**Tác giả**: Matthew Chun, Robert Clarke, Benjamin J. Cairns, David Clifton, Tingting Zhu et al. (University of Oxford & China Kadoorie Biobank Collaborative Group)  
**Tạp chí**: *Journal of the American Medical Informatics Association (JAMIA)*, 28(8), 2021, 1719–1727  

---

## 1. Quần thể nghiên cứu & Nguồn dữ liệu
- Dựa trên **China Kadoorie Biobank (CKB)**: Nghiên cứu đoàn hệ tiến cứu trên 503,842 người trưởng thành không có tiền sử đột quỵ từ 10 khu vực tại Trung Quốc (2004–2008), theo dõi trong 9 năm.
- Gồm 143 chỉ số nguy cơ (nhân khẩu học, chế độ ăn, hoạt động thể chất, đo đạc sinh hiệu).

## 2. Phương pháp & Mô hình
- Đánh giá khả năng dự đoán đột quỵ 9 năm và theo các khoảng thời gian (0–3 năm, 3–6 năm, 6–9 năm).
- So sánh các mô hình Cox regression, Random Survival Forest (RSF), Logistic Regression (LR), SVM, Gradient Boosted Trees (GBT), và Multilayer Perceptron (MLP) với thang điểm Framingham (FSRP 2017).
- Đề xuất kiến trúc **Ensemble**: Kết hợp mô hình Cox truyền thống và GBT bằng cây quyết định (Decision Tree) để chọn lựa mô hình tối ưu khi có sự bất đồng giữa 2 mô hình.

## 3. Kết quả chính
- **GBT đạt độ phân biệt (Discrimination) và độ chuẩn xác (Calibration) tốt nhất**:
  - AUROC dự đoán 9 năm: **0.833** (Nam), **0.836** (Nữ).
  - Độ chuẩn xác $\chi^2$: 44 (Nam), 47 (Nữ) — vượt trội so với FSRP ($\chi^2 > 5500$).
- **Phương pháp Ensemble**:
  - Đạt độ chính xác (Accuracy) 76% (Nam), 80% (Nữ), Sensitivity 76%–80%, Specificity 76%–81%.
  - Giúp cải thiện phân loại ở nhóm bệnh nhân nguy cơ cao mà vẫn giữ tính giải thích của mô hình Cox cho 95% bệnh nhân thông thường.