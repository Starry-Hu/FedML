import torch.utils.data as data

import pandas as pd
import torch

from sklearn.model_selection import train_test_split


class Cervical_truncated(data.Dataset):

    def __init__(self, datadir, dataidxs=None, train=True):
        self.datadir = datadir
        self.dataidxs = dataidxs
        self.train = train
        self.feature_name = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                             'Smokes', 'Smokes (years)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
                             'IUD', 'IUD (years)', 'STDs', 'STDs (number)',
                             'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
                             'STDs: Time since last diagnosis', 'Biopsy']
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        df = pd.read_csv(self.datadir, na_values=['?'])
        df = df[self.feature_name]
        # 数据清洗：用众数填充nan
        for index in df.columns:
            df[index].fillna(df[index].mode()[0], inplace=True)

        # 特征缩放
        # df.iloc[:, :-1] = (df - df.mean()) / df.std()

        # 70%train, 30%test
        df_list = train_test_split(df, test_size=0.3)
        data_list = [tuple] * 2

        # 处理数据得到x, y
        for index, df_one in enumerate(df_list):
            target = pd.DataFrame(df_one['Biopsy'])
            data_list[index] = {
                'X': torch.tensor(df_one.drop('Biopsy', axis=1).values).float(),
                'y': torch.tensor(target['Biopsy'].values).float()
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
            return data['X'], data['y']

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)
