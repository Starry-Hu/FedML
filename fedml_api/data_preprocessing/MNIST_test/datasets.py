import torch.utils.data as data
from torchvision.datasets import MNIST

import numpy as np


# from torchvision import transforms
# def get_train_data(batch_size):
#     train_loader = data.DataLoader(
#         MNIST('mnist_data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ])),
#         batch_size=batch_size, shuffle=True)
#
#     test_loader = data.DataLoader(
#         MNIST('mnist_data', train=False, transform=transforms.Compose([
#             transforms.ToTensor()
#         ])),
#         batch_size=batch_size, shuffle=True)
#

# 将mnist10数据进行截断
class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = mnist_dataobj.train_data
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)
        else:
            data = mnist_dataobj.data
            target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
