import torch
import torch.nn as nn
import torch.optim as optim


class DenseModel(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.01, bias=True):
        super(DenseModel, self).__init__()
        self.classifier = nn.Sequential(  # 快速构建神经网络
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias),
        )
        self.is_debug = False
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, weight_decay=0.01, lr=learning_rate)

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()  # detach切断反向传播，为啥需要用detach？

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        # 将数据转成相应的tensor形式
        x = torch.tensor(x, requires_grad=True).float()  # 告诉自动梯度机制记录这个张量上的计算
        grads = torch.tensor(grads).float()
        # 获得输出、(输出)反向传播、获得梯度
        output = self.classifier(x)
        output.backward(gradient=grads)
        x_grad = x.grad.numpy()

        # 梯度反向传播计算后之后，使用step进行所有参数的单次优化
        self.optimizer.step()
        # 清空当前梯度
        self.optimizer.zero_grad()

        return x_grad


class LocalModel(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(LocalModel, self).__init__()
        self.classifier = nn.Sequential(  # 相比DenseModel，加上了ReLU进行了修正
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.LeakyReLU()
        )
        self.output_dim = output_dim
        self.is_debug = False
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.parameters(), momentum=0.9, weight_decay=0.01, lr=learning_rate)

    def forward(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.forward")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def predict(self, x):
        if self.is_debug: print("[DEBUG] DenseModel.predict")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def backward(self, x, grads):
        if self.is_debug: print("[DEBUG] DenseModel.backward")

        x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_output_dim(self):
        return self.output_dim
