import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import argparse
from pathlib import Path
import logging


def loadData(outputPath):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=outputPath, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=outputPath, train=False, download=True, transform=transform
    )

    with h5py.File(f"Dataset.h5", "w") as hf:
        hf.create_dataset("trainset-data", data=trainset.data)
        hf.create_dataset("trainset-labels", data=trainset.targets)
        hf.create_dataset("testset-data", data=testset.data)
        hf.create_dataset("testset-labels", data=testset.targets)


logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser(description="CiFar10 Data Loading")

parser.add_argument(
    "--output-path",
    type=str,
    dest="outputPath",
    help="Path of the local file where the cifar data should be written.",
)

args = parser.parse_args()

Path(args.outputPath).parent.mkdir(parents=True, exist_ok=True)

with open(args.outputPath, "w") as output1_file:
    loadData(output1_file)
