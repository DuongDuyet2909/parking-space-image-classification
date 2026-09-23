import os
import pickle
import argparse
import json
import platform
import sklearn
from preprocessing import load_features

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # prepare data
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_dir, "clf-data", "clf-data")
    parser = argparse.ArgumentParser(description="Train SVM for cropped parking-space images")
    parser.add_argument('--data-dir', default=input_dir)
    parser.add_argument('--model', default=os.path.join(project_dir, 'model.pkl'))
    parser.add_argument('--report', default=os.path.join(project_dir, 'results', 'metrics.json'))
    args = parser.parse_args()
    input_dir = args.data_dir
    categories = ["empty", "not_empty"]

    data = []
    labels = []
    for category_idx, category in enumerate(categories):
        category_dir = os.path.join(input_dir, category)
        if not os.path.isdir(category_dir):
            parser.error(f'Missing class directory: {category_dir}')
        for file in sorted(os.listdir(category_dir)):
            img_path = os.path.join(category_dir, file)
            if not os.path.isfile(img_path) or not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue
            data.append(load_features(img_path)) # cùng tiền xử lý với predict.py
            labels.append(category_idx) # category_idx là nhãn của ảnh, 0 cho empty và 1 cho not_empty

    data = np.asarray(data) # asarray chuyển đổi danh sách data thành mảng numpy
    labels = np.asarray(labels)
    if any(np.sum(labels == i) < 10 for i in range(len(categories))):
        parser.error('Need at least 10 images per class for test split and five-fold CV.')

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=42) # giữ tỷ lệ lớp; chưa phải chia theo camera/thời gian


    # train classifier
    clsf = SVC(kernel='linear', C=1.0) # C điều chỉnh mức phạt lỗi; sẽ được GridSearchCV chọn lại

    parameters = {'C': [1, 10, 100, 1000]} # gamma không có tác dụng với kernel tuyến tính

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

    model_path = os.path.abspath(args.model)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(best_estimator, file) # lưu mô hình ngay trong cùng thư mục với file Python

    print("Model đã được lưu tại:", model_path)
    report = {
        'evaluation': 'stratified random image split; not camera/time-independent',
        'seed': 42,
        'classes': categories,
        'class_counts': {name: int(np.sum(labels == i)) for i, name in enumerate(categories)},
        'train_count': len(y_train), 'test_count': len(y_test),
        'best_params': grid_search.best_params_,
        'cv_accuracy': float(grid_search.best_score_),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'classification_report': classification_report(y_test, y_pred, target_names=categories, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print('Metrics saved to:', args.report)

    # Dự đoán nhãn cho một ảnh bất kỳ trong tập kiểm tra
    sample_image = X_test[0].reshape(15, 15, 3) # reshape ảnh từ 1D trở lại 3D với kích thước 15x15 và 3 kênh màu (RGB)
    sample_image_label = y_test[0] # nhãn thực tế của ảnh mẫu
    predicted_label = best_estimator.predict(X_test[0].reshape(1, -1)) # dự đoán nhãn của ảnh mẫu
    print("Sample image label:", sample_image_label)
    print("Predicted label:", categories[predicted_label[0]])


if __name__ == "__main__":
    main()
