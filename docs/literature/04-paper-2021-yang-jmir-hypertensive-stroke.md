# Paper Summary: Yang et al. (JMIR Med Inform 2021)

**Tiêu đề**: Accurate Prediction of Stroke for Hypertensive Patients Based on Medical Big Data and Machine Learning Algorithms: Retrospective Study  
**Tác giả**: Yujie Yang, Jing Zheng, Zhenzhen Du, Ye Li, Yunpeng Cai (Shenzhen Institute of Advanced Technology, CAS)  
**Tạp chí**: *JMIR Medical Informatics* (2021;9(11):e30277)  

---

## 1. Mục tiêu & Tập bệnh nhân
- Xây dựng mô hình dự đoán nguy cơ đột quỵ 3 năm cho bệnh nhân cao huyết áp dựa trên dữ liệu EMR lớn từ Nền tảng Dữ liệu Y tế Thâm Quyến (Shenzhen Health Information Big Data Platform).
- Thẩm định trên 57,671 bệnh nhân cao huyết áp (trong đó có 9,421 ca đột quỵ phát sinh trong 3 năm theo dõi).

## 2. Đặc trưng đầu vào & Phương pháp
- Trích xuất 49 đặc trưng từ EMR bao gồm nhân khẩu học, triệu chứng lịch sử, lối sống và **đặc trưng xu hướng đa thời gian (multitemporal trend characteristics)** như biến thiên huyết áp tâm thu/tâm trương (`SBP_mean`, `DBP_delta_mean`, `PPD_mean`).
- So sánh 4 thuật toán ML: Logistic Regression, SVM, Random Forest, XGBoost.
- Đối chứng hiệu năng với các thang điểm nguy cơ truyền thống: Framingham Stroke Risk Profile (FSRP) và Chinese Multiprovincial Cohort Study (CMCS).

## 3. Kết quả cốt lõi
- **XGBoost đạt hiệu năng vượt trội nhất**:
  - **AUC**: **0.9220** (so với Random Forest 0.8956, SVM 0.8898, LR 0.8544).
  - **Accuracy**: 0.8478 | **Recall**: 0.8512 | **F1-score**: 0.8319.
- **So sánh với thang điểm truyền thống (trên tập kiểm chứng độc lập 632 mẫu)**:
  - XGBoost 3-year AUC: **0.7956**
  - FSRP 10-year AUC: **0.6219** | FSRP 3-year AUC: **0.6199** | CMCS 10-year AUC: **0.6248**
- **Đặc trưng quan trọng nhất**: Tuổi (`age`), Số năm bị cao huyết áp (`Years_after_hypertension`), và các chỉ số xu hướng biến thiên huyết áp (`DBP_mean`, `PPD_delta_mean`).