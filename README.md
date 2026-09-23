# Parking Space Image Classification

Phân loại **ảnh đã cắt của một ô đỗ xe** thành `empty` (trống) hoặc
`not_empty` (có phương tiện), sử dụng SVM tuyến tính.
Chương trình chưa phát hiện vị trí ô đỗ trong ảnh toàn cảnh hoặc video.

## Phương pháp

Ảnh → chuyển RGB → resize 15 × 15 → flatten thành 675 giá trị → SVM.
Train và predict dùng chung `preprocessing.py`.
GridSearchCV tìm C trong [1, 10, 100, 1000] với 5 fold trên tập train.
Kernel tuyến tính không sử dụng gamma.

## Cài đặt

Môi trường đã kiểm thử train và predict: Python 3.12.14 trên Windows. Chạy trong thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.

## Dự đoán

Chuẩn bị một ảnh cắt của ô đỗ xe và truyền đường dẫn:

```powershell
.\.venv\Scripts\python.exe predict.py --image "C:\duong-dan\anh-o-do.jpg"
```

Ảnh và nhãn sẽ hiện trong cửa sổ Matplotlib. Lưu kết quả mà không mở cửa sổ:

```powershell
.\.venv\Scripts\python.exe predict.py --image "C:\duong-dan\anh-o-do.jpg" --no-show --output results/prediction.png
```

Model mặc định là `model.pkl` cạnh script; dùng `--model` để chọn model khác.
Metadata của model hiện có ghi scikit-learn 1.9.0; requirements ghim phiên bản này.
Đường dẫn tương đối do người dùng truyền được tính từ thư mục terminal.
Chỉ tải file pickle từ nguồn tin cậy.

## Dataset và huấn luyện

Do kích thước lớn, dataset không được đẩy trực tiếp lên repository này. Bạn có thể tải dataset về máy tại: 
👉 [Tải Dataset tại đây](https://drive.google.com/file/d/12zOiqLWDRUM7CB6s4rDQssheCIiOhoQz/view?usp=sharing)

*(Ghi chú: Ảnh trong thư mục `assets` chỉ là ảnh chụp kết quả minh họa, không phải ảnh đầu vào để huấn luyện).*

Chuẩn bị dữ liệu:
```text
clf-data/clf-data/
├── empty/
│   └── ...jpg
└── not_empty/
    └── ...jpg
```

```powershell
.\.venv\Scripts\python.exe Image_classification.py
```

Hoặc dùng dataset ở vị trí khác:

```powershell
.\.venv\Scripts\python.exe Image_classification.py --data-dir "C:\duong-dan\dataset"
```

Thư mục được truyền phải chứa trực tiếp hai thư mục lớp.
Lệnh train ghi đè `model.pkl` và `results/metrics.json` mặc định;
dùng `--model` và `--report` để chọn nơi lưu khác.
Báo cáo gồm số ảnh, tham số tốt nhất, accuracy CV/test, precision/recall/F1,
confusion matrix và phiên bản Python/NumPy/scikit-learn.

## Đánh giá và giới hạn

Model trong repo đã được train lại bằng pipeline hiện tại trên dataset cục bộ.
Kết quả chi tiết: [results/metrics.json](results/metrics.json).

| Chỉ số | Kết quả |
|---|---:|
| Số ảnh mỗi lớp | 3.045 |
| Train / test | 4.872 / 1.218 |
| C tốt nhất | 10 |
| Accuracy CV trung bình | 99,61% |
| Accuracy test | 100% |

Confusion matrix (hàng thật, cột dự đoán):

| | empty | not_empty |
|---|---:|---:|
| empty | 609 | 0 |
| not_empty | 0 | 609 |

**100% chỉ là kết quả trên lần chia ngẫu nhiên này, không phải độ chính xác
trong mọi điều kiện thực tế.** Kiểm tra SHA-256 không thấy file trùng byte giữa
train và test (6.090 hash khác nhau); điều này không loại trừ ảnh gần giống,
cùng camera hoặc các khung hình sát nhau. Chưa có metadata để đánh giá theo nhóm.

- Chia 80% train / 20% test với stratification, seed 42.
- Chỉ tìm tham số trên train; test dùng đánh giá sau khi chọn mô hình.
- Thứ tự nhãn của confusion matrix: empty, not_empty; hàng là nhãn thật,
  cột là dự đoán.
- Chia ngẫu nhiên theo ảnh chưa kiểm soát ảnh trùng hoặc ảnh gần nhau từ
  cùng camera/video. Kết quả không chứng minh khả năng tổng quát hóa sang
  camera, ngày hay bãi đỗ khác.
- Khi có metadata, cần chia theo nhóm camera/ngày/video cho cả test và CV.
- Resize 15 × 15 làm mất chi tiết; SVM trên pixel có thể nhạy với ánh sáng,
  góc nhìn và cách cắt ảnh. Chưa có thí nghiệm chứng minh tốt hơn HOG/CNN.
- Đặc biệt kiểm tra lỗi có xe nhưng báo trống: ô hàng not_empty, cột empty.
- Hai ảnh demo dưới đây minh họa giao diện, không thay cho thống kê test.

## Demo

![Dự đoán vị trí trống](assets/prediction-empty.png)
![Dự đoán vị trí có xe](assets/prediction-not-empty.png)

## Cấu trúc

```text
.
├── assets/                  # Ảnh chụp demo
├── Image_classification.py  # Train và xuất báo cáo
├── predict.py               # Dự đoán qua dòng lệnh
├── preprocessing.py         # Tiền xử lý dùng chung
├── model.pkl
├── requirements.txt
├── tests/
└── results/metrics.json     # Sinh sau khi train
```

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
