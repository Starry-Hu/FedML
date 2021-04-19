import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

from fedml_api.contribution.horizontal import my_model_trainer_classification

import numpy as np
import shap


class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #
    #     self.conv_layers = nn.Sequential(
    #         nn.Conv2d(1, 10, kernel_size=5),
    #         nn.MaxPool2d(2),
    #         nn.ReLU(),
    #         nn.Conv2d(10, 20, kernel_size=5),
    #         nn.Dropout(),
    #         nn.MaxPool2d(2),
    #         nn.ReLU(),
    #     )
    #     self.fc_layers = nn.Sequential(
    #         nn.Linear(320, 50),
    #         nn.ReLU(),
    #         nn.Dropout(),
    #         nn.Linear(50, 10),
    #         nn.Softmax(dim=1)
    #     )
    #
    # def forward(self, x):
    #     x = self.conv_layers(x)
    #     x = x.view(-1, 320)
    #     x = self.fc_layers(x)
    #     return x

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.max_pooling1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout1 = nn.Dropout()
        self.max_pooling2 = nn.MaxPool2d(2, stride=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.max_pooling1(self.conv1(x)))
        x = self.relu2(self.max_pooling2(self.dropout1(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.dropout2(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return self.softmax(x)
        # return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output = F.softmax(output,dim=1)
        # loss = F.nll_loss(output.log(), target)
        # loss = criterion(output, target)
        try:
            model.softmax
            criterion = nn.NLLLoss().to(device)
            loss = criterion(output.log(), target)
        except AttributeError:
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # criterion = nn.CrossEntropyLoss().to(device)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            try:
                model.softmax
                criterion = nn.NLLLoss().to(device)
                loss = criterion(output.log(), target).item()
            except AttributeError:
                criterion = nn.CrossEntropyLoss().to(device)
                loss = criterion(output, target).item()

            # output = F.softmax(output, dim=1)
            # test_loss += F.nll_loss(output.log(), target).item()  # sum up batch loss
            # test_loss += criterion(output, target).item()  # sum up batch loss
            test_loss += loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    X_display, y_display = shap.datasets.adult(display=True)


    batch_size = 128
    num_epochs = 3
    device = torch.device('cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # since shuffle=True, this is a random sample of test data
    # 获取到第一个batch
    batch = next(iter(test_loader))  # tensor 128(batch size)
    images, _ = batch

    print(images.shape)  # torch.Size([128, 1, 28, 28])

    background = images[:100]  # 训练数据的一部分  torch.Size([100, 1, 28, 28])
    test_images = images[100:103]  # 用来测试 torch.Size([3, 1, 28, 28])，取三个图片进行计算测试

    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)  # list 10, 每个里面[3, 1, 28, 28]

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]  # 对应显示的阴影图片个数，list 10, 每个里面[3, 28, 28, 1]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)  # [3, 28, 28, 1]),总共三个，每个里面[28,28,1]。对应上shap_numpy的每个元素的大小
    # plot the feature attributions  对四个预测的每个类的解释，从左到右排列，有序解释0-9类
    shap.image_plot(shap_numpy, -test_numpy)  # second is pixel_values