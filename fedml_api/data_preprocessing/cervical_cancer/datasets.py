import torch.utils.data as data

import pandas as pd
import torch

from sklearn.model_selection import train_test_split


class Cervical_truncated(data.Dataset):

    def __init__(self, datadir, dataidxs=None, train=True):
        self.datadir = datadir
        self.dataidxs = dataidxs
        self.train = train
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        df = pd.read_csv(self.datadir)
        df = df[['code', 'name', 'turnoverratio']]
        # 80%train, 20%test
        df_list = train_test_split(df, test_size=0.3)
        data_list = [tuple] * 2
        # 处理数据得到x, y
        for index, df_one in enumerate(df_list):
            target = pd.DataFrame(df_one['Biopsy'])
            data_list[index] = {
                'X': torch.tensor(df_one.drop('Biopsy', axis=1).values),
                'y': torch.tensor(target['Biopsy'].values)
            }
        # 判断训练数据/测试数据
        if self.train:
            data = data_list[0]
        else:
            data = data_list[1]
        # 返回具体某个客户端的数据
        if self.dataidxs is not None:
            return data['X'][self.dataidxs], data['y'][self.dataidxs]
        else:
            return data['X'][self.dataidxs], data['y'][self.dataidxs]

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)
