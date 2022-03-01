import cv2
import numpy as np
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import h5py
import json
import torchvision
from pathlib import Path
import os

if __name__ == "__main__":

    logging.getLogger("Data Loader").setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s |  %(name)s[%(process)d] | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--image-path", type=str, dest="image", help="S3 row data directory"
    )
    parser.add_argument(
        "-l", "--output", type=str, dest="out", help="S3 labels directory"
    )

    args = parser.parse_args()

    image_path, output_dir = args.image, args.out

    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    face_image_cv2 = cv2.imread(image_path)
    filename = "data/im.jpg"

    cv2.imwrite(filename, face_image_cv2)
    print(face_image_cv2.shape)
    ## print(face_image_cv2.tolist())
    print(len(face_image_cv2.tolist()))
    dirs = {
        "data": filename,
    }

    Path(output_dir).write_text(json.dumps(dirs))
