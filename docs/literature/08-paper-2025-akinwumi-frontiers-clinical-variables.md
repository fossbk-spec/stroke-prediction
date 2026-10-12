# Paper Summary: Akinwumi et al. (Front. Neurol. 2025)

**Tiêu đề**: Evaluating machine learning models for stroke prediction based on clinical variables  
**Tác giả**: Patrick O. Akinwumi, Stephen Ojo, Thomas I. Nathaniel et al. (Greenville Health System & Frontiers in Neurology)  
**Tạp chí**: *Frontiers in Neurology*, 16:1668420 (Published September 2025)  

---

## 1. Dữ liệu & Xử lý mất cân bằng
- Sử dụng bộ dữ liệu Kaggle Stroke Prediction (5,110 bản ghi).
- Tỷ lệ đột quỵ ban đầu chỉ 4.9% (249/5110). Áp dụng Random Over-Sampling để cân bằng tập Train.

## 2. Mô hình & Đánh giá
- Thử nghiệm 5 thuật toán: Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine (SVM), K-Nearest Neighbours (KNN) với Stratified 5-Fold Cross-Validation.
- **Kết quả Accuracy & ROC-AUC**:
  - Logistic Regression: **Accuracy 95.11%**, **ROC-AUC 0.8362**
  - Gradient Boosting: **Accuracy 94.90%**, **ROC-AUC 0.8236**
  - Random Forest: **Accuracy 94.95%**, **ROC-AUC 0.7923**

## 3. Thách thức lâm sàng cốt lõi (Recall Limitation)
- Mặc dù Accuracy và ROC-AUC rất cao, ma trận nhầm lẫn (Confusion Matrix) cho thấy hầu hết các mô hình đều thất bại trong việc phát hiện ca đột quỵ thực tế trên tập test (LR, SVM, RF đạt Recall = 0, Gradient Boosting & KNN chỉ phát hiện đúng 1 ca TP).
- **Kết luận**: Nhấn mạnh tầm quan trọng của việc không phụ thuộc vào Accuracy/ROC-AUC toàn cục mà phải tối ưu hóa Recall/Sensitivity và áp dụng Cost-Sensitive Learning trong dự đoán sự kiện hiếm y tế.