import os
import argparse
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread
from preprocessing import extract_features, to_rgb

def main():
    categories = ["empty", "not_empty"]

    # Tải model đã huấn luyện
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_dir, "model.pkl")
    parser = argparse.ArgumentParser(description='Predict a cropped parking-space image')
    parser.add_argument('--image', required=True, help='Path to your input image')
    parser.add_argument('--model', default=model_path)
    parser.add_argument('--no-show', action='store_true', help='Do not open a GUI window')
    parser.add_argument('--output', help='Save the prediction figure to this path')
    args = parser.parse_args()
    model_path = args.model
    if not os.path.isfile(args.image):
        parser.error(f'Image not found: {args.image}')
    if not os.path.isfile(model_path):
        parser.error(f'Model not found: {model_path}. Train the model first.')

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    image_path = args.image
    image = imread(image_path)
    image_to_show = to_rgb(image)

    # Tiền xử lý giống hệt lúc train
    features = extract_features(image).reshape(1, -1)
    if getattr(model, 'n_features_in_', None) != features.shape[1]:
        parser.error('Model does not match RGB 15x15 preprocessing; retrain it.')

    predicted_index = model.predict(features)[0]

    print("Nhãn số:", predicted_index)
    print("Kết quả:", categories[predicted_index])

    # Hiển thị ảnh và kết quả dự đoán trong một cửa sổ.
    if args.no_show:
        plt.switch_backend('Agg')
    plt.figure(figsize=(8, 5))
    if image_to_show.ndim == 2:
        plt.imshow(image_to_show, cmap="gray")
    else:
        plt.imshow(image_to_show[:, :, :3])
    plt.title(f"Kết quả dự đoán: {categories[predicted_index]}")
    plt.axis("off")
    plt.tight_layout()
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        plt.savefig(args.output, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
