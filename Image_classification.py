import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# prepare data
project_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(project_dir, "clf-data", "clf-data")
categories = ["empty", "not_empty"]

data = []
labels = []
for category_idx, category in enumerate(categories):
    category_dir = os.path.join(input_dir, category)
    for file in os.listdir(category_dir):
        img_path = os.path.join(category_dir, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten()) # flatten nghia là chuyển đổi ảnh 2D thành 1D
        labels.append(category_idx) # category_idx là nhãn của ảnh, 0 cho empty và 1 cho not_empty

data = np.asarray(data) # asarray chuyển đổi danh sách data thành mảng numpy
labels = np.asarray(labels)

# train / test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42) # giải thích: test_size=0.2 nghĩa là 20% dữ liệu sẽ được sử dụng để kiểm tra, random_state=42 để đảm bảo kết quả có thể tái tạo


# train classifier
clsf = SVC(kernel='linear', C=1.0, random_state=42) # classifier SVM với kernel tuyến tính, C=1.0 là tham số điều chỉnh độ phạt của lỗi, random_state=42 để đảm bảo kết quả có thể tái tạo

parameters = {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]} # gamma là tham số điều chỉnh độ cong của hàm kernel, C là tham số điều chỉnh độ phạt của lỗi

grid_search = GridSearchCV(clsf, parameters, cv=5, n_jobs=-1) # grid search để tìm các tham số tối ưu cho classifier

grid_search.fit(X_train, y_train) # fit dữ liệu huấn luyện vào grid search để tìm các tham số tối ưu

# test performance
best_estimator = grid_search.best_estimator_ # best_estimator_ là mô hình tốt nhất được tìm thấy từ grid search

y_pred = best_estimator.predict(X_test) # dự đoán nhãn của dữ liệu kiểm tra

# print performance metrics
print("Best parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(best_estimator, file) # lưu mô hình ngay trong cùng thư mục với file Python

print("Model đã được lưu tại:", model_path)

# Dự đoán nhãn cho một ảnh bất kỳ trong tập kiểm tra
sample_image = X_test[0].reshape(15, 15, 3) # reshape ảnh từ 1D trở lại 3D với kích thước 15x15 và 3 kênh màu (RGB)
sample_image_label = y_test[0] # nhãn thực tế của ảnh mẫu   
predicted_label = best_estimator.predict(X_test[0].reshape(1, -1)) # dự đoán nhãn của ảnh mẫu
print("Sample image label:", sample_image_label)
print("Predicted label:", categories[predicted_label[0]])
