from typing import List

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


class Preprocess(nn.Module):
    def __init__(self, input_shape: List[int]):
        super(Preprocess, self).__init__()
        self.input_shape = tuple(input_shape)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(
            input=x,
            size=self.input_shape[2:],
        )
        x = x / 255.0
        x = (x - self.mean) / self.std

        return x


if __name__ == "__main__":
    input_shape = [1, 3, 448, 448]
    output_onnx_file = "preprocessing.onnx"
    model = Preprocess(input_shape=input_shape)

    torch.onnx.export(
        model,
        torch.randn(input_shape),
        output_onnx_file,
        opset_version=20,
        input_names=["input_rgb"],
        output_names=["output_preprocessing"],
        dynamic_axes={
            "input_rgb": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
        },
    )

    model_onnx = onnx.load(output_onnx_file)
    model_simplified, _ = simplify(model_onnx)
    onnx.save(model_simplified, output_onnx_file)
