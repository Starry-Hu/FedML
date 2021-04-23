import torch.utils.data as data

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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
        df = df.apply(pd.to_numeric)
        # 数据清洗，填充nan。连续变量用中位数，离散型变量用0/1
        df['Number of sexual partners'] = df['Number of sexual partners'].fillna(
            df['Number of sexual partners'].median())
        df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
        df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
        df['Smokes'] = df['Smokes'].fillna(1)
        df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
        # df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
        df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
        df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(
            df['Hormonal Contraceptives (years)'].median())
        df['IUD'] = df['IUD'].fillna(0)
        df['IUD (years)'] = df['IUD (years)'].fillna(0)
        df['STDs'] = df['STDs'].fillna(1)
        df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
        # df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
        # df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(
        #     df['STDs:cervical condylomatosis'].median())
        # df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(
        #     df['STDs:vaginal condylomatosis'].median())
        # df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(
        #     df['STDs:vulvo-perineal condylomatosis'].median())
        # df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
        # df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(
        #     df['STDs:pelvic inflammatory disease'].median())
        # df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
        # df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(
        #     df['STDs:molluscum contagiosum'].median())
        # df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
        # df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
        # df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
        # df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
        df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(
            df['STDs: Time since first diagnosis'].median())
        df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(
            df['STDs: Time since last diagnosis'].median())
        # 类别变量one-got编码填充nan（此处忽略）
        # df = pd.get_dummies(data=df, columns=['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
        #                                       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Citology',
        #                                       'Schiller'])

        # 75%train, 25%test, 控制每次输出一样
        train_ds, test_ds = train_test_split(df, test_size=0.25, random_state=2021, shuffle=False)

        # 正则化，处理数据得到x, y
        minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_y = torch.tensor(train_ds['Biopsy'].values).float()
        train_X = torch.tensor(minmax_scale.fit_transform(train_ds.drop('Biopsy', axis=1))).float()
        test_y = torch.tensor(test_ds['Biopsy'].values).float()
        test_X = torch.tensor(minmax_scale.fit_transform(test_ds.drop('Biopsy', axis=1))).float()

        # 判断训练数据/测试数据
        if self.train:
            data = train_X
            target = train_y
        else:
            data = test_X
            target = test_y
        # 返回具体某个客户端的数据
        if self.dataidxs is not None:
            return data[self.dataidxs], target[self.dataidxs]
        else:
            return data, target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)
