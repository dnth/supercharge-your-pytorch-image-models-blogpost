import numpy as np
import onnxruntime as ort
from PIL import Image


def read_image(filename: str):
    img = Image.open(filename)
    img = np.array(img)
    img = img.astype(np.uint8)
    return img


onnx_filename = "eva02_large_patch14_448_pre_simplified.onnx"
providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 8589934592,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_force_sequential_engine_build": False,
            "trt_max_partition_iterations": 10000,
            "trt_min_subgraph_size": 1,
            "trt_builder_optimization_level": 5,
            "trt_timing_cache_enable": True,
        },
    ),
]
session = ort.InferenceSession(onnx_filename, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run(
    [output_name], {input_name: read_image("beignets-task-guide.png")}
)[0]

print(output.shape)
print(output)

import time

# Run benchmark
num_images = 100
start = time.perf_counter()
for i in range(num_images):
    output = session.run(
        [output_name], {input_name: read_image("beignets-task-guide.png")}
    )[0]
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(
    f"Onnxruntime builtin transforms: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}"
)
