# Kaggle Stroke Prediction Dataset Specification

**Nguồn**: Kaggle (fedesoriano)  
**Quy mô**: 5,110 bản ghi (bệnh nhân), 12 thuộc tính  
**Giấy phép**: Educational / Research License  

---

## 1. Cấu trúc các thuộc tính (Attribute Information)

| STT | Tên thuộc tính | K kiểu dữ liệu | Mô tả chi tiết |
|-----|----------------|----------------|----------------|
| 1 | `id` | Identifier | Mã định danh duy nhất của bệnh nhân |
| 2 | `gender` | Categorical | Giới tính ("Male", "Female", "Other") |
| 3 | `age` | Continuous | Tuổi bệnh nhân (năm) |
| 4 | `hypertension` | Binary | 0: Không cao huyết áp, 1: Có cao huyết áp |
| 5 | `heart_disease` | Binary | 0: Không có bệnh tim, 1: Có bệnh tim |
| 6 | `ever_married` | Binary | Trạng thái kết hôn ("No", "Yes") |
| 7 | `work_type` | Categorical | Loại việc làm ("children", "Govt_job", "Never_worked", "Private", "Self-employed") |
| 8 | `Residence_type` | Categorical | Khu vực sinh sống ("Rural", "Urban") |
| 9 | `avg_glucose_level` | Continuous | Mức đường huyết trung bình trong máu (mg/dL) |
| 10 | `bmi` | Continuous | Chỉ số khối cơ thể Body Mass Index ($kg/m^2$) |
| 11 | `smoking_status` | Categorical | Trạng thái hút thuốc ("formerly smoked", "never smoked", "smokes", "Unknown") |
| 12 | `stroke` | Binary (Target) | 1: Đã từng bị đột quỵ, 0: Chưa từng bị đột quỵ |

---

## 2. Thách thức chất lượng dữ liệu
- **Mất cân bằng lớp (Class Imbalance)**: 
  - Lớp 0 (Non-stroke): 4,861 mẫu (95.1%)
  - Lớp 1 (Stroke): 249 mẫu (4.9%)
- **Dữ liệu khuyết thiếu (Missing Values)**:
  - Thuộc tính `bmi`: Có 201 bản ghi thiếu giá trị (`N/A`).
  - Thuộc tính `smoking_status`: 1,544 bản ghi mang giá trị `Unknown` (chiếm ~30.2%).