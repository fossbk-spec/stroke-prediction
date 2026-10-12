# Paper Summary: Soladoye et al. (Int J Med Inform 2025)

**Tiêu đề**: Machine learning techniques for stroke prediction: A systematic review of algorithms, datasets, and regional gaps  
**Tác giả**: Afeez Adekunle Soladoye, Nicholas Aderinto, David B. Olawade et al.  
**Tạp chí**: *International Journal of Medical Informatics* 203 (2025) 106041  

---

## 1. Quy mô & Tiêu chuẩn Tổng quan Hệ thống (Systematic Review)
- Tiến hành theo quy chuẩn **PRISMA**, tổng hợp 58 nghiên cứu từ 2013 đến 2024 across 5 cơ sở dữ liệu (PubMed, Google Scholar, Lens, ResearchGate, Semantic Scholar).
- **3 Mục tiêu dự đoán chính**:
  1. Dự đoán nguy cơ phát sinh đột quỵ (Stroke Occurrence): 52 nghiên cứu (62.7%)
  2. Dự đoán tiên lượng/kết cục (Stroke Outcome): 19 nghiên cứu (22.9%)
  3. Phân loại phân nhóm đột quỵ (Stroke Type Classification): 12 nghiên cứu (14.4%)

## 2. Nguồn dữ liệu & Thuật toán
- **Nguồn dữ liệu**: Hồ sơ sức khỏe điện tử EHR (57.8%), Hình ảnh y tế MRI/CT (25.3%), Tín hiệu sinh học EEG/ECG (16.9%).
- **Thuật toán nổi bật**: Các phương pháp Ensemble Learning (Random Forest, XGBoost) đạt độ chính xác cao nhất cho dữ liệu EHR bảng (90.4%–97.8%), trong khi Deep Learning (CNN, LSTM) chiếm ưu thế ở dữ liệu hình ảnh và tín hiệu.

## 3. Khoảng trống địa lý & Rào cản triển khai
- **Khoảng trống khu vực nghiêm trọng**: Mặc dù Châu Phi có tỷ lệ tử vong do đột quỵ cao nhất thế giới (chiếm 87% ca tử vong ở các nước đang phát triển), số nghiên cứu từ khu vực này ít hơn 4 bài báo.
- **Rào cản**: Phụ thuộc lớn vào dữ liệu mở Kaggle chưa rõ nguồn gốc dịch tễ, thiếu quy trình thẩm định lâm sàng thực tế (Real-world clinical validation).