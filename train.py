import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


data_dir = 'transformed_data'


def load_split_train_test(datadir, valid_size=.2, batch_size=8):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
        ])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
        ])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)
    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

train_loader, test_loader = load_split_train_test(data_dir, .2, 8)
num_classes = train_loader.dataset.classes

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5

for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}')
    train_loss = 0.0
    valid_loss = 0.0

    correct = 0.0
    total = 0.0

    model.train()
    for train_step, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        # accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # log step
        if train_step % 20 == 0:
            print(f'Train step {train_step}/{len(train_loader)}, train loss {train_loss / (train_step + 1)}, accuracy {correct / total:.4}')
        
    # validate-the-model
    correct = 0.0
    total = 0.0
    print('Validation...')
    model.eval()
    for valid_step, (data, target) in enumerate(valid_loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        # accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # log step
        if valid_step % 20 == 0:
            print(f'Valid step {valid_step}/{len(valid_loader)}, valid loss {valid_loss / (valid_step + 1)}, accuracy {correct / total:.4}')
