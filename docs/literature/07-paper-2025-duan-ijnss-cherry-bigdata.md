# Paper Summary: Duan et al. (Int J Nurs Sci 2025)

**Tiêu đề**: Development and validation of a stroke risk prediction model using regional healthcare big data and machine learning  
**Tác giả**: Yunxia Duan, Rui Wang, Yumei Sun, Hongyu Sun et al. (Peking University)  
**Tạp chí**: *International Journal of Nursing Sciences* 12 (2025) 558–565  

---

## 1. Thiết kế nghiên cứu & Dữ liệu
- Nghiên cứu đoàn hệ hồi cứu dựa trên nền tảng dữ liệu y tế CHERRY (Chinese Electronic health Records Research in Yinzhou, Ningbo) giai đoạn 2015–2021.
- Bao gồm **92,172 người trưởng thành**, ghi nhận 436 ca đột quỵ mắc mới (tỷ lệ 474/100,000 person-years).
- Cân bằng dữ liệu bằng SMOTE và Random Undersampling tạo tỷ lệ 1:1 (20,000 mẫu).

## 2. Lựa chọn đặc trưng & Mô hình
- Sàng lọc 13 đặc trưng có khả năng can thiệp/thay đổi được (modifiable risk factors) và lâm sàng cốt lõi: Cao huyết áp, Tuổi, Đái tháo đường, Huyết áp tâm thu (SBP), Vòng eo, HDL-C, Đường huyết lúc đói (FBG), Hoạt động thể chất, BMI, LDL-C, Cholesterol toàn phần, Chế độ ăn, Tiền sử gia đình.
- So sánh Logistic Regression (LR) với 4 thuật toán ML: Decision Tree (DT), Random Forest (RF), XGBoost, và Back Propagation Neural Network (BPNN).

## 3. Kết quả chính
- **Random Forest (RF) đạt hiệu năng cao nhất**:
  - **AUC Train / Val**: **0.988 / 0.980** (so với LR 0.777/0.779, BPNN 0.921/0.918, XGBoost 0.962/0.958).
  - **Accuracy**: 0.919 | **Precision**: 0.912 | **Recall**: 0.929 | **F1 Score**: 0.921.
- **Xác định điểm cắt phân tầng rủi ro (Risk Stratification Threshold)**:
  - Sử dụng Youden's Index xác định **Cutoff = 0.789**.
  - Bệnh nhân có xác suất $\ge 0.789$ được phân loại vào nhóm nguy cơ cao (High-risk) phục vụ can thiệp điều dưỡng/y tế cộng đồng.