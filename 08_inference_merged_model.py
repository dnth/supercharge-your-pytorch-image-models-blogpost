import time
from urllib.request import urlopen

import cupy as cp
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from imagenet_classes import IMAGENET2012_CLASSES

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)


def read_image(image: Image.Image):
    image = image.convert("RGB")
    img_numpy = np.array(image).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy


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

session = ort.InferenceSession("merged_model_compose.onnx", providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run([output_name], {input_name: read_image(img)})

# print(output[0])

# Check the output
output = torch.from_numpy(output[0])
print(output.shape)


top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

im_classes = list(IMAGENET2012_CLASSES.values())
class_names = [im_classes[i] for i in top5_class_indices[0]]

# Print class names and probabilities
for name, prob in zip(class_names, top5_probabilities[0]):
    print(f"{name}: {prob:.2f}%")

num_images = 1000
start = time.perf_counter()
for i in range(num_images):
    output = session.run([output_name], {input_name: read_image(img)})
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(f"Onnxruntime TensorRT: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}")
