# Paper Summary: Hong et al. (JAMA 2023)

**Tiêu đề**: Predictive Accuracy of Stroke Risk Prediction Models Across Black and White Race, Sex, and Age Groups  
**Tác giả**: Hong, Chuan; Pencina, Michael J.; Wojdyla, Daniel M.; Hall, Jennifer L.; Judd, Suzanne E.; Cary, Michael; Engelhard, Matthew M.; Berchuck, Samuel; Xian, Ying; D’Agostino, Ralph; Howard, George; Kissela, Brett; Henao, Ricardo  
**Tạp chí**: *JAMA* (2023; 329(4): 306) — DOI: https://doi.org/10.1001/jama.2022.24683  
**Nhóm phân loại**: Dữ liệu cohort  
**Citekey**: `Hong2023Race` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Abstract only (PMC10408266 full text not retrievable via Europe PMC API)

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- N = 62,482 người ≥45 tuổi (Framingham Offspring, ARIC, MESA, REGARDS), 29% người da đen.
- Tỷ lệ dương tính: Không báo cáo trong phần đã trích xuất.
- Thiết kế: cohort hồi cứu gộp.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Yếu tố nguy cơ tự báo cáo và của pooled cohort equations.
- Xử lý dữ liệu / mất cân bằng: Không báo cáo trong phần đã trích xuất.
- **Đánh giá:** nguy cơ 10 năm, phân biệt và hiệu chuẩn theo chủng tộc, giới, tuổi.

## 3. Thuật toán & Kết quả thực nghiệm
- Framingham stroke, REGARDS self-report, pooled cohort equations, 2 thuật toán ML mới.
- C index: Framingham 0.72, REGARDS 0.73, PCE 0.72; ML không cải thiện có ý nghĩa.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- Điểm mạnh: JAMA, nhiều cohort, đánh giá công bằng.
- Phát hiện: mọi mô hình phân biệt kém hơn ở người da đen. Bài học cho N22: báo cáo hiệu năng theo nhóm.
- Hạn chế: chỉ trích ở mức Abstract (chưa có PDF).

## Trích dẫn nguyên văn (Ground truth)
> "The combined study sample included 62 482 participants" (Abstract); "existing stroke-specific risk prediction models and novel machine learning techniques did not significantly improve discriminative accuracy for new-onset stroke compared with the pooled cohort equations" (Abstract, Conclusions); "All algorithms exhibited worse discrimination in Black individuals than in White individuals" (Abstract)
