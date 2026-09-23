import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb

categories = ["empty", "not_empty"]

# Tải model đã huấn luyện
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, "model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Thay bằng đường dẫn ảnh cần dự đoán
image_path = os.path.join(
    project_dir,
    "clf-data",
    "clf-data",
    "not_empty",
    "00000000_00000005.jpg"
)

image = imread(image_path)
image_to_show = image.copy()

# Nếu ảnh grayscale, chuyển thành RGB
if image.ndim == 2:
    image = gray2rgb(image)

# Nếu ảnh PNG có 4 channel RGBA, bỏ alpha
if image.shape[2] == 4:
    image = image[:, :, :3]

# Tiền xử lý giống hệt lúc train
image = resize(image, (15, 15))
features = image.flatten().reshape(1, -1)

predicted_index = model.predict(features)[0]

print("Nhãn số:", predicted_index)
print("Kết quả:", categories[predicted_index])

# Hiển thị ảnh và kết quả dự đoán trong một cửa sổ.
plt.figure(figsize=(8, 5))
if image_to_show.ndim == 2:
    plt.imshow(image_to_show, cmap="gray")
else:
    plt.imshow(image_to_show[:, :, :3])
plt.title(f"Kết quả dự đoán: {categories[predicted_index]}")
plt.axis("off")
plt.tight_layout()
plt.show()
