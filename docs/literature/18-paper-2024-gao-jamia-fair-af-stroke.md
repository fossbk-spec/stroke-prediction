# Paper Summary: Gao et al. (Journal of the American Medical Informatics Association 2024)

**Tiêu đề**: Fair prediction of 2-year stroke risk in patients with atrial fibrillation  
**Tác giả**: Gao, Jifan; Mar, Philip; Tang, Zheng-Zheng; Chen, Guanhua  
**Tạp chí**: *Journal of the American Medical Informatics Association* (2024; 31(12): 2820–2828) — DOI: https://doi.org/10.1093/jamia/ocae170  
**Nhóm phân loại**: Dữ liệu EMR lâm sàng lớn  
**Citekey**: `Gao2024Fair` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Abstract only (PMC11631105 full text not retrievable via Europe PMC API)

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N: Không báo cáo trong phần đã trích xuất (All of Us Research Program, bệnh nhân rung nhĩ).
- Thiết kế: EHR hồi cứu.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Biến của CHADS2 và CHA2DS2-VASc.
- Xử lý dữ liệu / mất cân bằng: Không báo cáo trong phần đã trích xuất.
- **Chia dữ liệu:** hold-out test theo thời gian, tăng tỷ lệ người da đen; tuning set đại diện → rủi ro **thấp**.

## 3. Thuật toán & Kết quả thực nghiệm
- LightGBM + chiến lược giảm chênh lệch (ngưỡng theo nhóm).
- AUROC từ gần 0.70 (thang điểm gốc) lên trên 0.80.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: diễn giải đầy đủ từng biến dự báo.
- Điểm mạnh: ngưỡng riêng theo nhóm để đạt công bằng — gợi ý trực tiếp cho bước chọn ngưỡng của N22.
- Hạn chế: chỉ quần thể rung nhĩ; trích ở mức Abstract.

## Trích dẫn nguyên văn (Ground truth)
> "significant improvements were achieved by modeling their predictors using machine learning models (Area Under the Receiver Operating Characteristic curve from near 0.70 to above 0.80)" (Abstract, Results)
