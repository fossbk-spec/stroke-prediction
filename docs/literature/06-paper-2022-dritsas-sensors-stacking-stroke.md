# Paper Summary: Dritsas & Trigka (Sensors 2022)

**Tiêu đề**: Stroke Risk Prediction with Machine Learning Techniques  
**Tác giả**: Elias Dritsas, Maria Trigka (University of Patras, Greece)  
**Tạp chí**: *Sensors* 2022, 22(13), 4670  

---

## 1. Dữ liệu & Tiền xử lý
- Sử dụng bộ dữ liệu Kaggle Stroke Prediction (lọc 3,254 người tham gia > 18 tuổi).
- Áp dụng kỹ thuật **SMOTE** để cân bằng lớp mẫu đột quỵ (Stroke) và không đột quỵ (Non-stroke), tạo ra tập dữ liệu cân bằng 6,148 mẫu.

## 2. Kiến trúc mô hình Stacking
- **Mô hình cơ sở (Base Classifiers)**: Naive Bayes, Random Forest, J48 (C4.5 Decision Tree), RepTree.
- **Mô hình siêu cấp (Meta-Classifier)**: Logistic Regression.
- Đánh giá bằng kiểm thử chéo 10 lần (10-fold Cross-Validation) trên phần mềm WEKA.

## 3. Kết quả thực nghiệm
- **Mô hình Stacking đạt hiệu năng cao nhất toàn diện**:
  - **AUC**: **98.9%** (0.989)
  - **Accuracy**: **98.0%**
  - **Precision**: **97.4%** | **Recall**: **97.4%** | **F-measure**: **97.4%**
- **Xếp hạng tầm quan trọng đặc trưng (Feature Importance)**:
  - Biến Tuổi (`age`) và Chỉ số BMI (`bmi`) là hai yếu tố có ảnh hưởng quyết định nhất tới nguy cơ đột quỵ.