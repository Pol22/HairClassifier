import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def load_split_train_test(datadir, test_size=.2, batch_size=8):
    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomAffine(20, scale=(1.0, 1.3)),
         transforms.RandomHorizontalFlip(),
        ])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(),
        ])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(test_size * num_train))
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
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)

        self.fc1 = nn.Linear(32 * 14 * 14, 48)
        self.fc2 = nn.Linear(48, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    data_dir = 'transformed_data'
    batch_size = 16
    test_size = 0.15
    num_epochs = 100
    log_freq = 20

    train_loader, test_loader = load_split_train_test(data_dir, test_size, batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training device: {device}')

    model = Net()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}')
        train_loss = 0.0
        test_loss = 0.0

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
            if train_step % log_freq == 0 or train_step == len(train_loader) - 1: 
                print(f'Train step {train_step}/{len(train_loader)}, train loss {train_loss / (train_step + 1):1.4}, accuracy {correct / total:.4}')

        correct = 0.0
        total = 0.0
        print('Testing...')
        model.eval()
        for test_step, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            # accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # log step
            if test_step % log_freq == 0 or test_step == len(test_loader) - 1:
                print(f'Test step {test_step}/{len(test_loader)}, test loss {test_loss / (test_step + 1):1.4}, accuracy {correct / total:.4}')

        torch.save(
            {
                'model_state_dict': model.state_dict(),
            },
            f'model_{epoch}.pt')

