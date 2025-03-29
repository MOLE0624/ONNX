import unittest

import numpy as np
import onnxruntime as ort
from mutil.ops.normalization import NormMethod, normalize

from ..layers.norm import NormLayerONNX


class TestNormLayerShapeONNX(unittest.TestCase):
    def setUp(self):
        self.input_shape: tuple = (1, 3, 64, 64)
        self.input_data: np.ndarray = (
            np.random.rand(*self.input_shape).astype(np.float32) * 255.0
        )
        self.min_vals: tuple = (0.0, 0.0, 0.0)
        self.max_vals: tuple = (255.0, 255.0, 255.0)
        self.mean_vals: tuple = (0.5, 0.5, 0.5)
        self.std_vals: tuple = (0.1, 0.1, 0.1)

    def test_minmax_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.MINMAX,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        normalized_data = norm_layer.run(self.input_data)
        self.assertEqual(normalized_data[0].shape, tuple(self.input_shape))

    def test_zscore_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.ZSCORE,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        normalized_data = norm_layer.run(self.input_data)
        self.assertEqual(normalized_data[0].shape, tuple(self.input_shape))

    def test_compound_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.COMPOUND,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        normalized_data = norm_layer.run(self.input_data)
        self.assertEqual(normalized_data[0].shape, tuple(self.input_shape))


class TestNormLayerDataONNX(unittest.TestCase):
    def setUp(self):
        self.input_shape: tuple = (1, 3, 64, 64)
        self.input_data: np.ndarray = np.random.rand(*self.input_shape).astype(
            np.float32
        )
        self.min_vals: tuple = (0.0, 0.0, 0.0)
        self.max_vals: tuple = (255.0, 255.0, 255.0)
        self.mean_vals: tuple = (0.5, 0.5, 0.5)
        self.std_vals: tuple = (0.1, 0.1, 0.1)

    def test_minmax_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.MINMAX,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        expected_output = normalize(
            self.input_data,
            NormMethod.MINMAX,
            self.min_vals,
            self.max_vals,
            self.mean_vals,
            self.std_vals,
        )
        expected_output = np.array(expected_output)
        onnx_output = norm_layer.run(self.input_data)[0]
        np.testing.assert_allclose(onnx_output, expected_output, rtol=1e-5, atol=1e-5)

    def test_zscore_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.ZSCORE,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        expected_output = normalize(
            self.input_data,
            NormMethod.ZSCORE,
            self.min_vals,
            self.max_vals,
            self.mean_vals,
            self.std_vals,
        )
        onnx_output = norm_layer.run(self.input_data)[0]
        np.testing.assert_allclose(onnx_output, expected_output, rtol=1e-5, atol=1e-5)

    def test_compound_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.COMPOUND,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        expected_output = normalize(
            self.input_data,
            NormMethod.COMPOUND,
            self.min_vals,
            self.max_vals,
            self.mean_vals,
            self.std_vals,
        )
        onnx_output = norm_layer.run(self.input_data)[0]
        np.testing.assert_allclose(onnx_output, expected_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
