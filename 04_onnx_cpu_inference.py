import time
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image

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
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_numpy = (img_numpy - mean) / std
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)
    return img_numpy


# Create ONNX Runtime session with CPU provider
onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(onnx_filename, providers=["CPUExecutionProvider"])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], {input_name: transforms_numpy(img)})[0]

# Run benchmark
num_images = 10
start = time.perf_counter()
for i in range(num_images):
    output = session.run([output_name], {input_name: transforms_numpy(img)})[0]
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(f"Onnxruntime CPU: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}")
