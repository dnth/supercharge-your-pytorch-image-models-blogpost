from urllib.request import urlopen

import timm
import torch
from PIL import Image

from imagenet_classes import IMAGENET2012_CLASSES

if __name__ == "__main__":
    model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    model = timm.create_model(model_name, pretrained=True).eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )

    with torch.inference_mode():
        output = model(transforms(img).unsqueeze(0))

    top5_probabilities, top5_class_indices = torch.topk(
        output.softmax(dim=1) * 100, k=5
    )
    im_classes = list(IMAGENET2012_CLASSES.values())
    class_names = [im_classes[i] for i in top5_class_indices[0]]

    print("Top 5 predictions:")
    for name, prob in zip(class_names, top5_probabilities[0]):
        print(f"  {name}: {prob:.2f}%")
