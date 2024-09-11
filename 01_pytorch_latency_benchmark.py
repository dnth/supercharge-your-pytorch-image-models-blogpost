import time
from urllib.request import urlopen

import timm
import torch
from PIL import Image

model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
model = timm.create_model(model_name, pretrained=True).eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)


def run_benchmark(model, device, num_images=10):
    model = model.to(device)

    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(num_images):
            input_tensor = transforms(img).unsqueeze(0).to(device)
            model(input_tensor)
        end = time.perf_counter()

    ms_per_image = (end - start) / num_images * 1000
    fps = num_images / (end - start)

    print(f"PyTorch model on {device}: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}")


if __name__ == "__main__":
    run_benchmark(model, "cpu")
    run_benchmark(model, "cuda")
