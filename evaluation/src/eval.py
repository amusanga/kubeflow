import cv2
import logging
import argparse
import torch
import numpy as np
import json
from torchvision import models, transforms
from pathlib import Path
import os
from PIL import Image
import subprocess

transfomer = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":

    logging.getLogger("Data Encoding").setLevel(logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--image-path", type=str, dest="data", help="row data directory"
    )
    parser.add_argument(
        "-o", "--output-path", type=str, dest="out", help="row data directory"
    )
    args = parser.parse_args()

    with open(args.data) as json_file:
        data = json.load(json_file)

    image_path, output_dir = args.data, args.out

    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    input_image = Image.open(image_path).convert("RGB")

    input_tensor = transfomer(input_image)
    input_batch = input_tensor.unsqueeze(0)

    model = models.resnet50(pretrained=True)

    model.eval()
    output = model(input_batch)
    _, predicted = torch.max(output.data, 1)
    print(predicted[0])

    encoded_image = cv2.imencode(".jpeg", input_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    output = {"prediction": predicted, "image": encoded_image}

    Path(output_dir).write_text(json.dumps(output))
