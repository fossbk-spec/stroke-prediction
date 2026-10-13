# Paper Summary: Wallace et al. (BMC Medical Research Methodology 2023)

**Tiêu đề**: Use and misuse of random forest variable importance metrics in medicine: demonstrations through incident stroke prediction  
**Tác giả**: Wallace, Meredith L.; Mentch, Lucas; Wheeler, Bradley J.; Tapia, Amanda L.; Richards, Marc; Zhou, Siyu; Yi, Lixia; Redline, Susan; Buysse, Daniel J.  
**Tạp chí**: *BMC Medical Research Methodology* (2023; 23(1)) — DOI: https://doi.org/10.1186/s12874-023-01965-x  
**Nhóm phân loại**: Phương pháp luận  
**Citekey**: `Wallace2023VIMP` (xem `evisyn/evidence_matrix.json`, `evisyn/references.bib`)  
**Vị trí trích dẫn**: Full text PMC10280951: Abstract

---

## 1. Mục tiêu & Tập bệnh nhân (Cohort / Dataset)
- Tổng quan 50 bài dùng random forest + minh họa trên Sleep Heart Health Study (đột quỵ mới mắc 5 năm).
- N minh họa: Không báo cáo trong phần đã trích xuất.

## 2. Đặc trưng đầu vào & Phương pháp tiền xử lý
- Chức năng phổi, thuốc, yếu tố nguy cơ đo được.
- Không áp dụng mất cân bằng/chia dữ liệu (bài phương pháp).

## 3. Thuật toán & Kết quả thực nghiệm
- Random forest với OOB VIMP so với knockoff VIMP.
- Gần như mọi bài trong 50 bài có hạn chế đáng kể khi dùng VIMP; OOB VIMP đẩy hai biến chức năng phổi tương quan cao lên đầu.

## 4. Khả năng giải thích lâm sàng (XAI) & Đánh giá phê phán
- XAI: trọng tâm của bài.
- Bài học cho N22: không dùng OOB/impurity importance đơn lẻ; dùng SHAP/permutation theo nhóm biến và đối chiếu odds ratio.

## Trích dẫn nguyên văn (Ground truth)
> "We first evaluate current VIMP practices through an in-depth literature review of 50 recent random forest manuscripts" (Abstract); "Nearly all papers in the literature review contained substantial limitations in their use of VIMPs" (Abstract)
