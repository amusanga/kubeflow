import torch
import torchvision

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
