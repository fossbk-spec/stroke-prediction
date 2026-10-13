# Paper Summary: Melnykova et al. (Scientific Reports 2025)

**Tiêu đề**: Machine learning for stroke prediction using imbalanced data  
**Tác giả**: Melnykova, Nataliia; Patereha, Yurii; Skopivskyi, Stepan; Farion, Mykola; Fedushko, Solomia; Drohomyretska, Khrystyna  
**Tạp chí**: *Scientific Reports* (2025; 15(1)) — DOI: https://doi.org/10.1038/s41598-025-01855-w  
**Nhóm phân loại**: Dữ liệu mở Kaggle  
**Citekey**: `Melnykova2025Imbalanced` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC12484691: Abstract; Methods (grid search); Results para 34; Table 4

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 4,981 dòng, 11 cột.
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: benchmark trên dữ liệu công khai.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- 11 cột dạng Kaggle stroke.
- Xử lý dữ liệu thiếu: Không báo cáo trong phần đã trích xuất.
- **Mất cân bằng:** SMOTE; so sánh với không dùng SMOTE (Bảng 4).
- **Chia dữ liệu & rò rỉ:** chia 80-20; thứ tự SMOTE so với bước chia không nêu rõ → rủi ro **chưa xác định**.

## 3. Thuật toán & Kết quả thực nghiệm
- Nhiều mô hình; RF tốt nhất, tối ưu bằng grid search.
- RF: precision, recall, F1 90%, accuracy 90%; RF trên dữ liệu không cân bằng: accuracy 96%.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: không.
- Điểm mạnh: minh họa rõ accuracy 96% gây hiểu lầm trên dữ liệu lệch.
- Hạn chế: không kiểm định ngoài; mô tả chưa đủ để tái lập thứ tự resampling.

## Trích dẫn nguyên văn (Ground truth)
> "achieving precision, recall, and F1-score levels of 90%, along with an accuracy of 90%" (Abstract); "This approach demonstrated the model's high accuracy of 96%, underscoring the impracticality of using accuracy as the sole metric" (Abstract); "our dataset has 4981 rows and 11 columns" (Methods)
