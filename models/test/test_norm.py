import unittest

import numpy as np
import onnxruntime as ort

from ..layers.norm import NormLayerONNX, NormMethod


def numpy_normalization(
    input_data, method, min_vals, max_vals, mean_vals=None, std_vals=None
):
    if method == NormMethod.MINMAX:
        return (input_data - np.array(min_vals)[None, :, None, None]) / (
            np.array(max_vals)[None, :, None, None]
            - np.array(min_vals)[None, :, None, None]
        )
    elif method == NormMethod.ZSCORE:
        return (input_data - np.array(mean_vals)[None, :, None, None]) / np.array(
            std_vals
        )[None, :, None, None]
    elif method == NormMethod.COMPOUND:
        minmax = numpy_normalization(input_data, NormMethod.MINMAX, min_vals, max_vals)
        return (minmax - np.array(mean_vals)[None, :, None, None]) / np.array(std_vals)[
            None, :, None, None
        ]
    else:
        raise ValueError("Unknown normalization method")


def test_normalization(method):
    input_shape = [1, 3, 64, 64]
    input_data = np.random.rand(*input_shape).astype(np.float32) * 255.0

    min_vals = [0.0, 0.0, 0.0]
    max_vals = [255.0, 255.0, 255.0]
    mean_vals = [0.5, 0.5, 0.5]
    std_vals = [0.1, 0.1, 0.1]

    norm_layer = NormLayerONNX(
        input_shape=input_shape,
        method=method,
        min_vals=min_vals,
        max_vals=max_vals,
        mean_vals=mean_vals,
        std_vals=std_vals,
    )
    norm_layer.save("normalization.onnx")

    normalized_numpy = numpy_normalization(
        input_data, method, min_vals, max_vals, mean_vals, std_vals
    )

    sess = ort.InferenceSession("normalization.onnx")
    onnx_output = sess.run(None, {sess.get_inputs()[0].name: input_data})[0]

    assert np.allclose(
        normalized_numpy, onnx_output, atol=1e-5
    ), "ONNX output does not match NumPy"
    print(f"Test passed for method {method}")


class TestNormLayerShapeONNX(unittest.TestCase):
    def setUp(self):
        self.input_shape = [1, 3, 64, 64]
        self.input_data = np.random.rand(*self.input_shape).astype(np.float32) * 255.0
        self.min_vals = [0.0, 0.0, 0.0]
        self.max_vals = [255.0, 255.0, 255.0]
        self.mean_vals = [0.5, 0.5, 0.5]
        self.std_vals = [0.1, 0.1, 0.1]

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
        self.input_shape = [1, 3, 64, 64]
        self.input_data = np.random.rand(*self.input_shape).astype(np.float32)
        self.min_vals = [0.0, 0.0, 0.0]
        self.max_vals = [255.0, 255.0, 255.0]
        self.mean_vals = [0.5, 0.5, 0.5]
        self.std_vals = [0.1, 0.1, 0.1]

    def test_minmax_normalization(self):
        norm_layer = NormLayerONNX(
            input_shape=self.input_shape,
            method=NormMethod.MINMAX,
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            mean_vals=self.mean_vals,
            std_vals=self.std_vals,
        )
        expected_output = numpy_normalization(
            self.input_data,
            NormMethod.MINMAX,
            self.min_vals,
            self.max_vals,
            self.mean_vals,
            self.std_vals,
        )
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
        expected_output = numpy_normalization(
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
        expected_output = numpy_normalization(
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
