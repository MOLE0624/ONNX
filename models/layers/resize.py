# ================================================================================
# File       : resize.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a layer for resizing images in ONNX format.
# Date       : 2025-03-29
# ================================================================================

from enum import Enum
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


class ResizeMethod(Enum):
    NEAR = "nearest"
    LINE = "linear"
    CUBE = "cubic"
    LANC = "lanczos"


class ResizeLayerONNX:
    def __init__(
        self,
        input_shape: list,
        target_shape: list,
        opset: int = 11,
        method: str = ResizeMethod.NEAR,
        crop: list = None,
        dynamic: bool = False,
    ):
        """Initialize input and target dimensions with support for dynamic shapes.

        Args:
            input_shape: List of [N, C, H, W] for input dimensions
            target_shape: List of [N, C, H, W] for target dimensions
            dynamic: Whether to support dynamic batch and channel dimensions
        """
        # Validate input shapes
        if len(input_shape) != 4 or len(target_shape) != 4:
            raise ValueError(
                "Input and target shapes must be lists of length 4 [N, C, H, W]"
            )

        # Validate target dimensions cannot be -1
        if target_shape[2] == -1 or target_shape[3] == -1:
            raise ValueError("Target height and width cannot be -1")

        self.N, self.C, self.H, self.W = input_shape
        _, _, self.target_height, self.target_width = target_shape
        self.opset = opset
        self.method = method
        self.crop = crop
        self.dynamic = dynamic

    def get_layer(
        self, input_name="input", output_name="output"
    ) -> Tuple[
        onnx.NodeProto, List[onnx.TensorProto], onnx.ValueInfoProto, onnx.ValueInfoProto
    ]:
        """Return the ONNX Resize layer as a node with initializers."""
        # Define names for required tensors
        roi_name = "roi"
        scales_name = "scales"

        # Create ROI tensor (required by Resize op) - must be a real tensor, not an empty string
        if self.crop:
            crop_ymin, crop_xmin, crop_ymax, crop_xmax = self.crop
            roi = np.array(
                [0, 0, crop_ymin, crop_xmin, crop_ymax, crop_xmax, 1, 1],
                dtype=np.float32,
            )
        else:
            # No cropping, use full input dimensions
            roi = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

        roi_initializer = helper.make_tensor(roi_name, TensorProto.FLOAT, [8], roi)

        # Calculate scale factors - this is more reliable across different ONNX runtimes
        h_scale = float(self.target_height) / float(self.H)
        w_scale = float(self.target_width) / float(self.W)
        scales = np.array([1.0, 1.0, h_scale, w_scale], dtype=np.float32)

        scales_initializer = helper.make_tensor(
            scales_name, TensorProto.FLOAT, [4], scales
        )

        # Resize node with proper ROI and scales
        resize_node = helper.make_node(
            "Resize",
            inputs=[input_name, roi_name, scales_name],  # Use the defined ROI tensor
            outputs=[output_name],
            mode=self.method.value,
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
            name="Resize" + self.method.name,
        )
        # print(resize_node)

        # Define input/output shapes
        if self.dynamic:
            # For dynamic shapes, we use None for batch and channel dimensions
            # but enforce the spatial dimensions (H, W) to match the constructor
            input_tensor = helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, ["N", "C", "H", "W"]
            )
            output_tensor = helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                [self.N, self.C, self.target_height, self.target_width],
            )
        else:
            # Fixed shapes
            input_tensor = helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                [
                    "N" if self.N == -1 else self.N,
                    "C" if self.C == -1 else self.C,
                    "H" if self.H == -1 else self.H,
                    "W" if self.W == -1 else self.W,
                ],
            )
            output_tensor = helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                [
                    "N" if self.N == -1 else self.N,
                    "C" if self.C == -1 else self.C,
                    self.target_height,
                    self.target_width,
                ],
            )

        return (
            resize_node,
            [roi_initializer, scales_initializer],
            input_tensor,
            output_tensor,
        )

    def get_model(self) -> onnx.ModelProto:
        """Return the entire ONNX model including input/output definitions."""
        input_name, output_name = "input", "output"
        resize_node, initializers, input_tensor, output_tensor = self.get_layer(
            input_name, output_name
        )

        # Create ONNX graph
        graph = helper.make_graph(
            nodes=[resize_node],
            name="ResizeGraph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        # Create ONNX model with appropriate opset
        model = helper.make_model(graph, producer_name="onnx-resize")
        model.opset_import[0].version = self.opset

        # Add metadata to help with debugging
        model.doc_string = f"Resize model: {'dynamic' if self.dynamic else 'fixed'} shape, target: {self.target_height}x{self.target_width}"

        # Check the model
        onnx.checker.check_model(model)

        return model

    def save(self, filename="resize.onnx"):
        """Save the ONNX model to a file."""
        model = self.get_model()
        onnx.save(model, filename)
        print(f"Model saved to {filename}")
        return model

    def run(self, input_data: np.ndarray):
        """Run inference using ONNX Runtime."""
        # Convert input to float32 if needed
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Get model
        model = self.get_model()

        # Create session and run inference
        session = ort.InferenceSession(model.SerializeToString())
        output_name = session.get_outputs()[0].name
        input_name = session.get_inputs()[0].name

        return session.run([output_name], {input_name: input_data})


# Example Usage
if __name__ == "__main__":
    # Example with fixed shape using nearest interpolation
    print("===============================================")
    print("Creating fixed shape model with NearestNeighbor interpolation...")
    print("===============================================")
    resize_layer = ResizeLayerONNX(
        input_shape=[-1, 3, 1080, 1920],
        target_shape=[100, 3, 640, 640],
        method=ResizeMethod.NEAR,
        dynamic=False,
    )

    # Save the ONNX model
    # print("Saving fixed shape model with Nearest interpolation...")
    # model = resize_layer.save("resize_fixed.onnx")

    # Run inference
    print("\nRunning inference (Fixed Shape - Nearest Interpolation)...")
    input_data = np.random.rand(10, 3, 1080, 1920).astype(np.float32)
    output = resize_layer.run(input_data)
    print(f"Input Shape: {input_data.shape} -> Output Shape: {output[0].shape}\n")

    # Example with dynamic shape using cubic interpolation
    print("===============================================")
    print("Creating dynamic shape model with Cubic interpolation...")
    print("===============================================")
    resize_layer_dynamic = ResizeLayerONNX(
        input_shape=[1, 3, 1080, 1920],  # Base shape for dynamic
        target_shape=[1, 3, 640, 640],  # Base shape for dynamic
        method=ResizeMethod.CUBE,
        dynamic=True,
    )

    # Save the dynamic ONNX model
    # print("Saving dynamic shape model with Cubic interpolation...")
    # dynamic_model = resize_layer_dynamic.save("resize_dynamic.onnx")

    # Test with different batch size for dynamic model
    print("\nRunning inference (Dynamic Shape - Cubic Interpolation)...")
    test_input = np.random.rand(1, 3, 1080, 1920).astype(np.float32)
    dynamic_output = resize_layer_dynamic.run(test_input)
    print(
        f"Input Shape: {test_input.shape} -> Output Shape: {dynamic_output[0].shape}\n"
    )
