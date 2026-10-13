- **High — `src/models.py:155`, `src/train.py:81`**: Bản sửa stacking còn thiếu: `cv=5` chia theo bản ghi, không nhận `groups`, nên cùng bệnh nhân có thể xuất hiện ở train và validation nội bộ.
  **Sửa:** tạo meta-feature OOF theo nhóm bệnh nhân, giữ preprocessing/SMOTE trong từng fold; thêm test stacking với bệnh nhân nhiều bản ghi.

- **High — `src/explainability.py:143`, `tests/test_explainability.py:38`**: CI Wald bỏ intercept, không phản ánh regularization/SMOTE và dùng one-hot đồng tuyến; cờ `significant` không có cơ sở suy luận thống kê, trong khi test chỉ kiểm tra CI bao quanh ước lượng.
  **Sửa:** bootstrap toàn bộ quy trình theo bệnh nhân hoặc dùng mô hình suy luận phù hợp trên dữ liệu gốc; kiểm chứng CI bằng trường hợp tham chiếu.

- **Medium — `src/preprocessing.py:282`**: SMOTE nội suy cả one-hot và biến nhị phân, tạo “bệnh nhân” có giới tính, nghề nghiệp hoặc bệnh nền dạng phân số.
  **Sửa:** dùng SMOTENC trước one-hot, hoặc đối chiếu với nhánh class-weight không tạo mẫu tổng hợp.

- **Medium — `src/models.py:74`**: `**overrides` trùng tham số mặc định gây `TypeError`; đã xác nhận với `get_model("random_forest", n_estimators=10)`.
  **Sửa:** gộp dictionary mặc định với overrides trước khi khởi tạo; thêm test override cho từng factory.

- **Medium — `src/data_loader.py:97`**: Schema bỏ qua nhãn null; dữ liệu được xác nhận hợp lệ nhưng sau đó lỗi khi `astype(int)` trong `split_features_target`.
  **Sửa:** từ chối nhãn thiếu ngay khi validation; thêm test nhãn null và chuỗi không chuyển đổi được.

- **Medium — `requirements.txt:5`, `.github/workflows/ci_cd_pipeline.yml:44`**: Dependencies chỉ có cận dưới, không có lockfile; cùng seed vẫn có thể cho kết quả khác hoặc CI hỏng khi thư viện thay đổi.
  **Sửa:** khóa bộ phiên bản đã kiểm chứng và dùng chung cho local/CI; lưu phiên bản môi trường cùng kết quả thí nghiệm.

- **Medium — `notebooks/03_model_training_and_evaluation.ipynb:33`** *(cả ba notebook)*: Cell khởi tạo dùng đường dẫn tương đối và `chdir`; chạy lại cell trên Colab/Kaggle có thể clone repo lồng nhau, đồng thời `check=False` che lỗi cài dependencies.
  **Sửa:** xác định thư mục repo tuyệt đối, bootstrap idempotent và dùng `sys.executable -m pip` với `check=True`.

- **Medium — `.github/workflows/ci_cd_pipeline.yml:69`**: Smoke test bỏ CV và stacking mặc định, không chạy notebook; cũng chưa có test tích hợp kiểm tra `--min-recall` được áp dụng xuyên suốt `run_experiment`.
  **Sửa:** thêm kiểm thử tích hợp nhẹ cho CLI/chính sách ngưỡng và thực thi code cells notebook trong CI.

- **Low — `src/eda.py:133`**: Khi không có nhóm glucose `<100`, nhóm đầu tiên còn lại bị dùng làm “normal”; đã xác nhận toàn bộ glucose=150 vẫn trả risk ratio=1.
  **Sửa:** lấy baseline bằng nhãn nhóm bình thường; trả NaN nếu nhóm này vắng mặt.

- **Low — `docs/DETAILED_EXAM_REPORT.md:178`**: Ví dụ gọi `verify_zero_data_leakage(train_data, train_data, test_data)` luôn thất bại vì train và validation giống nhau.
  **Sửa:** thay ví dụ bằng API hiện hành và tạo validation độc lập.

- **Low — `Makefile:9`, `Makefile:30`**: `make setup` không cài ruff nhưng `make lint` yêu cầu ruff trong `.venv`, nên quy trình này lỗi trên môi trường mới.
  **Sửa:** cài `requirements-dev.txt` trong setup hoặc thêm target setup-dev rõ ràng.

- **Low — `README.md:9`, `README.md:60`**: Hướng dẫn “offline” vẫn cần GitHub/PyPI; mô tả holdout bằng StratifiedGroupKFold khác code hiện dùng stratified `train_test_split` trên ID.
  **Sửa:** ghi đúng yêu cầu mạng và phân biệt thuật toán holdout với grouped CV.

Không thấy Critical; luồng holdout và chọn ngưỡng chính hợp lý, nhưng stacking theo nhóm và suy luận CI cần sửa trước khi tin vào kết luận khoa học.  
Không sửa tệp; các test đã chọn ngoài SHAP chạy qua, ba test SHAP bị chặn bởi cache Matplotlib trong sandbox chỉ đọc; Logistic Regression khớp README.