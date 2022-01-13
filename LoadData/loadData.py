import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    print(testset)
    with h5py.File(f"Dataset.h5", "w") as hf:
        hf.create_dataset("trainset-data", data=trainset)
        hf.create_dataset("trainset-labels", data=trainset.targets)
        hf.create_dataset("testset-data", data=testset)
        hf.create_dataset("testset-labels", data=testset.targets)


if __name__ == "__main__":
    main()
