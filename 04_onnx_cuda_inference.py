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


def transforms_numpy(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((448, 448), Image.BICUBIC)
    img_numpy = np.array(image).astype(np.float32) / 255.0
    img_numpy = img_numpy.transpose(2, 0, 1)
    mean = np.array([0.4815, 0.4578, 0.4082]).reshape(-1, 1, 1)
    std = np.array([0.2686, 0.2613, 0.2758]).reshape(-1, 1, 1)
    img_numpy = (img_numpy - mean) / std
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)
    return img_numpy


def transforms_cupy(image: Image.Image):
    # Convert image to RGB and resize
    image = image.convert("RGB")
    image = image.resize((448, 448), Image.BICUBIC)

    # Convert to CuPy array and normalize
    img_cupy = cp.array(image, dtype=cp.float32) / 255.0
    img_cupy = img_cupy.transpose(2, 0, 1)

    # Apply mean and std normalization
    mean = cp.array([0.4815, 0.4578, 0.4082], dtype=cp.float32).reshape(-1, 1, 1)
    std = cp.array([0.2686, 0.2613, 0.2758], dtype=cp.float32).reshape(-1, 1, 1)
    img_cupy = (img_cupy - mean) / std

    # Add batch dimension
    img_cupy = cp.expand_dims(img_cupy, axis=0)

    return img_cupy


# Create ONNX Runtime session with CPU provider
onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(onnx_filename, providers=["CUDAExecutionProvider"])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], {input_name: transforms_numpy(img)})[0]


# Check the output
output = torch.from_numpy(output)
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

im_classes = list(IMAGENET2012_CLASSES.values())
class_names = [im_classes[i] for i in top5_class_indices[0]]

# Print class names and probabilities
for name, prob in zip(class_names, top5_probabilities[0]):
    print(f"{name}: {prob:.2f}%")

# Run benchmark numpy
num_images = 100
start = time.perf_counter()
for i in range(num_images):
    output = session.run([output_name], {input_name: transforms_numpy(img)})[0]
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(
    f"Onnxruntime CUDA numpy transforms: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}"
)


# Run benchmark cupy
num_images = 100
start = time.perf_counter()
for i in range(num_images):
    img_cupy = transforms_cupy(img)
    output = session.run([output_name], {input_name: cp.asnumpy(img_cupy)})[0]
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(
    f"Onnxruntime CUDA cupy transforms: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}"
)
