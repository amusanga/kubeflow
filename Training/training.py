import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import h5py
import torchvision
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import logging

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser(description="CiFar10 Data Loading")

parser.add_argument(
    "--output-path",
    type=str,
    help="Path of the local file where the cifar data should be written.",
)
parser.add_argument(
    "--input-path",
    type=str,
    help="Path of the local file where the Output 1 data should be written.",
)

args = parser.parse_args()

OUTPUTDIR = args.output_path
INPUTDIR = args.input_path

Path(OUTPUTDIR).parent.mkdir(parents=True, exist_ok=True)

with h5py.File(INPUTDIR, "r") as file:
    trainset_data = np.array(file.get("trainset-data")).astype(np.float32)
    trainset_labels = np.array(file.get("trainset-labels")).astype(np.float32)
    testset_data = np.array(file.get("testset-data")).astype(np.float32)
    testset_labels = np.array(file.get("testset-labels")).astype(np.float32)

file.close()


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data, labels):
        "Initialization"
        self.labels = labels
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        X = self.data[index].reshape(3, 32, 32)
        y = self.labels[index]
        return X, y


def default_collate(batch):
    data, labels = zip(*batch)
    data, labels = np.array(data), np.array(labels)
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).long()
    return data, labels


trainData = Dataset(trainset_data, trainset_labels)

testData = Dataset(testset_data, testset_labels)

trainloader = torch.utils.data.DataLoader(
    trainData, batch_size=4, shuffle=True, num_workers=2, collate_fn=default_collate
)

testloader = torch.utils.data.DataLoader(
    testData, batch_size=4, shuffle=True, num_workers=2, collate_fn=default_collate
)

print("********************************")
print(trainloader)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

PATH = OUTPUTDIR + "/cifar_net.pth"
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
