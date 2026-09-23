import unittest
import numpy as np
from preprocessing import extract_features


class PreprocessingTests(unittest.TestCase):
    def test_grayscale_equals_rgb(self):
        gray = np.arange(400, dtype=np.uint8).reshape(20, 20)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        np.testing.assert_allclose(extract_features(gray), extract_features(rgb))

    def test_rgba_uses_rgb_channels(self):
        rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
        rgba = np.concatenate([rgb, np.full((20, 20, 1), 255, dtype=np.uint8)], axis=2)
        np.testing.assert_allclose(extract_features(rgb), extract_features(rgba))

    def test_shape_range(self):
        result = extract_features(np.full((30, 50, 3), 255, dtype=np.uint8))
        self.assertEqual(result.shape, (675,))
        np.testing.assert_allclose(result, 1.0)

    def test_invalid_channels(self):
        with self.assertRaises(ValueError):
            extract_features(np.zeros((20, 20, 2)))


if __name__ == '__main__':
    unittest.main()
