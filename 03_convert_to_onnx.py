import timm
import torch

model = timm.create_model(
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True
).eval()

onnx_filename = "eva02_large_patch14_448.onnx"
torch.onnx.export(
    model,
    torch.randn(1, 3, 448, 448),
    onnx_filename,
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
