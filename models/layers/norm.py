# ================================================================================
# File       : norm.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a normalization layer in ONNX format.
# Date       : 2025-03-29
# ================================================================================
from enum import Enum
from typing import List, Tuple

import numpy as np
import onnx
from mutil.ops.normalization import NormMethod
from onnx import TensorProto, helper


class NormLayerONNX:
    def __init__(
        self,
        input_shape: List[int],
        method: NormMethod = NormMethod.MINMAX,
        min_vals: List[float] = None,
        max_vals: List[float] = None,
        mean_vals: List[float] = None,
        std_vals: List[float] = None,
        dynamic: bool = False,
    ):
        """
        Initialize the normalization layer.

        Args:
            input_shape: List of [N, C, H, W] for input dimensions.
            method: Normalization method (MINMAX, ZSCORE, or COMPOUND).
            min_vals: Minimum values for each channel (for MINMAX).
            max_vals: Maximum values for each channel (for MINMAX).
            mean_vals: Mean values for each channel (for ZSCORE).
            std_vals: Standard deviation values for each channel (for ZSCORE).
            dynamic: Whether to support dynamic batch and channel dimensions.
        """
        if len(input_shape) != 4:
            raise ValueError("Input shape must be a list of length 4 [N, C, H, W]")

        self.N, self.C, self.H, self.W = input_shape
        self.method = method
        self.dynamic = dynamic

        # Handle scalar min/max values by expanding them to per-channel values
        if min_vals is not None and len(min_vals) == 1:
            min_vals = min_vals * self.C
        if max_vals is not None and len(max_vals) == 1:
            max_vals = max_vals * self.C
        if mean_vals is not None and len(mean_vals) == 1:
            mean_vals = mean_vals * self.C
        if std_vals is not None and len(std_vals) == 1:
            std_vals = std_vals * self.C

        self.min_vals = (
            np.array(min_vals, dtype=np.float32) if min_vals is not None else None
        )
        self.max_vals = (
            np.array(max_vals, dtype=np.float32) if max_vals is not None else None
        )
        self.mean_vals = (
            np.array(mean_vals, dtype=np.float32) if mean_vals is not None else None
        )
        self.std_vals = (
            np.array(std_vals, dtype=np.float32) if std_vals is not None else None
        )

    def get_layer(
        self, input_name="input", output_name="output"
    ) -> Tuple[
        onnx.NodeProto, List[onnx.TensorProto], onnx.ValueInfoProto, onnx.ValueInfoProto
    ]:
        """
        Return the ONNX normalization layer as a node with initializers.
        """
        nodes = []
        initializers = []

        # Split the input tensor into channels
        split_name = "split"
        split_node = helper.make_node(
            "Split",
            inputs=[input_name],
            outputs=[f"{split_name}_{i}" for i in range(self.C)],
            axis=1,  # Split along the channel dimension
            name="SplitInput",
            num_outputs=self.C,  # Explicitly set the number of outputs to C (number of channels)
        )
        nodes.append(split_node)

        # Apply normalization for each channel based on the selected method
        for i in range(self.C):
            norm_name = f"norm_{i}"
            param_1_name = f"param_1_{i}"
            param_2_name = f"param_2_{i}"
            if self.method == NormMethod.COMPOUND:
                param_3_name = f"param_3_{i}"
                param_4_name = f"param_4_{i}"

            if self.method == NormMethod.MINMAX:
                # For Min-Max normalization: (x - min) / (max - min)
                param_1 = self.min_vals[i]
                param_2 = self.max_vals[i] - self.min_vals[i]
            elif self.method == NormMethod.ZSCORE:
                # For Z-score normalization: (x - mean) / std
                param_1 = self.mean_vals[i]
                param_2 = self.std_vals[i]
            elif self.method == NormMethod.COMPOUND:
                # First apply Min-Max normalization
                param_1 = self.min_vals[i]
                param_2 = self.max_vals[i] - self.min_vals[i]
                # Then apply Z-score normalization to Min-Max result
                param_3 = self.mean_vals[i]
                param_4 = self.std_vals[i]

            # Create param_1 and param_2 tensors (min/max or mean/std)
            param_1_tensor = helper.make_tensor(
                param_1_name, TensorProto.FLOAT, [1], [param_1]
            )
            param_2_tensor = helper.make_tensor(
                param_2_name, TensorProto.FLOAT, [1], [param_2]
            )
            initializers.extend([param_1_tensor, param_2_tensor])

            # Normalize using the selected method

            if self.method == NormMethod.MINMAX or self.method == NormMethod.ZSCORE:
                norm_node = helper.make_node(
                    "Sub",
                    inputs=[f"{split_name}_{i}", param_1_name],
                    outputs=[f"norm_{i}_sub"],
                    name=f"Norm_{i}_Sub",
                )
                nodes.append(norm_node)

                norm_node = helper.make_node(
                    "Div",
                    inputs=[f"norm_{i}_sub", param_2_name],
                    outputs=[norm_name],
                    name=f"Norm_{i}_Div",
                )
                nodes.append(norm_node)

            elif self.method == NormMethod.COMPOUND:
                min_max_sub_node = helper.make_node(
                    "Sub",
                    inputs=[f"{split_name}_{i}", param_1_name],
                    outputs=[f"norm_{i}_minmax_sub"],
                    name=f"Norm_{i}_MinMaxSub",
                )
                nodes.append(min_max_sub_node)

                min_max_div_node = helper.make_node(
                    "Div",
                    inputs=[f"norm_{i}_minmax_sub", param_2_name],
                    outputs=[f"norm_{i}_minmax"],
                    name=f"Norm_{i}_MinMaxDiv",
                )
                nodes.append(min_max_div_node)

                param_3_tensor = helper.make_tensor(
                    param_3_name, TensorProto.FLOAT, [1], [param_3]
                )
                param_4_tensor = helper.make_tensor(
                    param_4_name, TensorProto.FLOAT, [1], [param_4]
                )
                initializers.extend([param_3_tensor, param_4_tensor])

                compound_sub_node = helper.make_node(
                    "Sub",
                    inputs=[f"norm_{i}_minmax", param_3_name],
                    outputs=[f"norm_{i}_compound_sub"],
                    name=f"Norm_{i}_CompoundSub",
                )
                nodes.append(compound_sub_node)

                compound_div_node = helper.make_node(
                    "Div",
                    inputs=[f"norm_{i}_compound_sub", param_4_name],
                    outputs=[norm_name],
                    name=f"Norm_{i}_CompoundDiv",
                )
                nodes.append(compound_div_node)

        # Concatenate the normalized channels back together
        concat_node = helper.make_node(
            "Concat",
            inputs=[f"norm_{i}" for i in range(self.C)],
            outputs=[output_name],
            axis=1,  # Concatenate along the channel dimension
            name="ConcatNormalized",
        )
        nodes.append(concat_node)

        # Define input/output shapes
        if self.dynamic:
            input_tensor = helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, ["N", "C", "H", "W"]
            )
            output_tensor = helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, ["N", "C", "H", "W"]
            )
        else:
            input_tensor = helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [self.N, self.C, self.H, self.W]
            )
            output_tensor = helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [self.N, self.C, self.H, self.W]
            )

        return (
            nodes,
            initializers,
            input_tensor,
            output_tensor,
        )

    def get_model(self) -> onnx.ModelProto:
        """
        Return the entire ONNX model including input/output definitions.
        """
        input_name, output_name = "input", "output"
        nodes, initializers, input_tensor, output_tensor = self.get_layer(
            input_name, output_name
        )

        # Create ONNX graph
        graph = helper.make_graph(
            nodes=nodes,
            name="NormalizationGraph",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        # Create ONNX model
        model = helper.make_model(graph, producer_name="onnx-normalization")
        onnx.checker.check_model(model)

        return model

    def save(self, filename="normalization.onnx"):
        """
        Save the ONNX model to a file.
        """
        model = self.get_model()
        onnx.save(model, filename)
        print(f"Model saved to {filename}")
        return model

    def run(self, input_data: np.ndarray):
        """
        Run inference using ONNX Runtime.
        """
        import onnxruntime as ort

        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        model = self.get_model()
        session = ort.InferenceSession(model.SerializeToString())
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return session.run([output_name], {input_name: input_data})


if __name__ == "__main__":
    # Example input shape (batch size, channels, height, width)
    input_shape = [
        1,
        3,
        64,
        64,
    ]  # For example, 1 image with 3 channels (RGB) and 64x64 resolution

    # Example data: Random image data for testing
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Example min, max, mean, and std values for each channel (3 channels)
    min_vals = [0.0, 0.0, 0.0]
    max_vals = [255.0, 255.0, 255.0]
    mean_vals = [0.5, 0.5, 0.5]
    std_vals = [0.1, 0.1, 0.1]

    # Choose the normalization method (e.g., MINMAX, ZSCORE, or COMPOUND)
    # method = NormMethod.MINMAX
    method = NormMethod.ZSCORE
    # method = NormMethod.COMPOUND

    # Create the normalization layer
    norm_layer = NormLayerONNX(
        input_shape=input_shape,
        method=method,
        min_vals=min_vals,
        max_vals=max_vals,
        mean_vals=mean_vals,
        std_vals=std_vals,
    )

    # Save the model to an ONNX file
    norm_layer.save("normalization.onnx")

    # Run the model (inference) with the example input data
    normalized_data = norm_layer.run(input_data)

    # Print the result
    print("Normalized data:")
    print(normalized_data[0].shape)
