from argparse import ArgumentParser
from typing import List

import numpy as np
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

        # self.register_buffer(
        #     "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # )
        # self.register_buffer(
        #     "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # )

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(
            input=x,
            size=self.input_shape[2:],
        )
        # x = x * (1.0 / 255.0)
        x = x / 255.0
        x = (x - self.mean) / self.std

        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--opset", type=int, default=20, help="onnx opset")
    parser.add_argument(
        "-s",
        "--input_shape",
        type=int,
        nargs=4,
        default=[1, 3, 448, 448],
        help="input shape",
    )

    args = parser.parse_args()

    MODEL = f"01_prep"
    OPSET = args.opset
    INPUT_SHAPE: List[int] = args.input_shape

    model = Preprocess(input_shape=INPUT_SHAPE)

    onnx_file = f"{MODEL}_{'_'.join(map(str, INPUT_SHAPE))}.onnx"
    x = torch.randn(INPUT_SHAPE)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=["input_rgb"],
        output_names=["output_prep"],
        dynamic_axes={
            "input_rgb": {
                2: "H",
                3: "W",
            },
        },
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
