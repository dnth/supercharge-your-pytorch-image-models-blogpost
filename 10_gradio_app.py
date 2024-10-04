import time
from urllib.request import urlopen

import gradio as gr
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from imagenet_classes import IMAGENET2012_CLASSES


def read_image(image: Image.Image):
    image = image.convert("RGB")
    img_numpy = np.array(image).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy


providers = ["CPUExecutionProvider"]

session = ort.InferenceSession("merged_model_compose.onnx", providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict(img):
    output = session.run([output_name], {input_name: read_image(img)})
    output = torch.from_numpy(output[0])

    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1), k=5)

    im_classes = list(IMAGENET2012_CLASSES.values())
    class_names = [im_classes[i] for i in top5_class_indices[0]]

    results = {
        name: float(prob) for name, prob in zip(class_names, top5_probabilities[0])
    }
    return results


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classification with ONNX TensorRT",
    description="Upload an image to classify it using the ONNX TensorRT model.",
)

if __name__ == "__main__":
    iface.launch()
