"""Shared train/inference preprocessing: RGB, 15x15, 675 features."""
import numpy as np
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize


def to_rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = gray2rgb(image)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = gray2rgb(image[:, :, 0])
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f'Expected grayscale, RGB or RGBA image; got {image.shape}')
    return image


def extract_features(image):
    features = resize(to_rgb(image), (15, 15)).flatten()
    if not np.isfinite(features).all():
        raise ValueError('Image contains NaN or infinity.')
    return features


def load_features(path):
    return extract_features(imread(path))
