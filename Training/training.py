import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import h5py

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# trainset_data, trainset_label = np.load(
#     "/home/aime/kubeflow/LoadData/trainset-data.npy"
# ), np.load("/home/aime/kubeflow/LoadData/trainset-labels.npy")
# testset_data, testset_label = np.load(
#     "/home/aime/kubeflow/LoadData/testset-data.npy"
# ), np.load("/home/aime/kubeflow/LoadData/testset-labels.npy")

filename = "/home/aime/kubeflow/LoadData/Dataset.h5"

with h5py.File(filename, "r") as file:
    trainset_data = np.array(file.get("trainset-data"))
    trainset_labels = np.array(file.get("trainset-labels"))
    testset_data = np.array(file.get("testset-data"))
    testset_labels = np.array(file.get("testset-labels"))

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


trainData = Dataset(trainset_data, trainset_labels)

testData = Dataset(testset_data, testset_labels)

trainloader = torch.utils.data.DataLoader(
    trainData, batch_size=4, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testData, batch_size=4, shuffle=True, num_workers=2
)


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


for epoch in range(2):  # loop over the dataset multiple times

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

PATH = "./cifar_net.pth"
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
