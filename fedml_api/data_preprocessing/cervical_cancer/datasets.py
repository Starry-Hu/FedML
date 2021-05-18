import torch.utils.data as data

import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def data_visualize(df,feature_name):
    # 绘制Smokes等离散变量与其相关连续变量的关系,Pie chart
    # 自定义格式
    # def my_fmt(x):
    #     '%1.1f%%'
    #     total = 858
    #     print(x)
    #     return '{:.1f}%\n({:.0f})'.format(x, total * x / 100)
    #
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax = [ax1, ax2, ax3, ax4]
    # labels = ['all nan', 'all not nan']
    # colors = ['#ff9999', '#ffcc99']
    # checkFeatures = [{'A': 'Smokes', 'B': 'Smokes (years)'},
    #                  {'A': 'Hormonal Contraceptives', 'B': 'Hormonal Contraceptives (years)'},
    #                  {'A': 'IUD', 'B': 'IUD (years)'},
    #                  {'A': 'STDs', 'B': 'STDs (number)'}]
    # for feature, features in enumerate(checkFeatures):
    #     df_bool1 = ((df[features['A']].isna()) & df[features['B']].isna())
    #     df_bool4 = (~(df[features['A']].isna()) & (~df[features['B']].isna()))
    #     count = [df_bool1.sum(), df_bool4.sum()]
    #
    #     ax[feature].pie(count, colors=colors, labels=labels, autopct=my_fmt, startangle=90)
    #     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    #     fig = plt.gcf()
    #     fig.gca().add_artist(centre_circle)
    #     ax[feature].axis('equal')
    #     if feature != 1:
    #         ax[feature].set_title('{} and {}'.format(features['A'], features['B']))
    #     else:
    #         ax[feature].set_title('HC and HC (years) for short')
    #     plt.tight_layout()
    # plt.show()

    # Visualize the number of missing
    # values as a bar chart
    # import missingno as msno
    # msno.bar(df)

    # scale = preprocessing.StandardScaler()
    # data = scale.fit_transform(df.drop('Biopsy', axis=1))
    # df = pd.DataFrame(data, index=list(range(data.shape[0])), columns=feature_name)
    import seaborn as sns
    # 描述各特征的分布密度图
    print("Density Plots");
    print()
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9),
          (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]
    i = 0
    for feature in df[feature_name]:
        print('*' * 100)
        sns.countplot(x=feature, data=df, ax=ax[i])  # 计数图
        # sns.distplot(df[feature], ax=ax[i])  # 数据分布密度图
        i+=1
    plt.show()



    x = ['Biopsy-0', 'Biopsy-1']
    y = [len(df[df["Biopsy"] == 0]), len(df[df["Biopsy"] == 1])]
    plt.bar(x, y, width=0.8)
    plt.ylabel("count")
    for a, b in zip(x, y):
        plt.text(a, b + 0.25, '%.0f' % b, ha="center", va="bottom", fontsize=12)
    # plt.savefig("./bar1.png")
    plt.show()
    # for a, b in zip(X, df['Bi']):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)

    # checking the patch order, not for final:
    # catp.ax.text(spot[0].get_x(), -3, spot[1][0][0]+spot[1][1][0])
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    # sns.countplot(x='Age', data=df, ax=ax1)
    # sns.countplot(x='Biopsy', data=df, ax=ax2)
    # sns.barplot(x='Age', y='Biopsy', data=df, ax=ax3)

    # Stratified
    facet = sns.FacetGrid(df, hue='Biopsy', aspect=4)
    facet.map(sns.kdeplot, 'Age', shade=True)
    facet.set(xlim=(0, df['Age'].max()))
    facet.add_legend()

    plt.show()


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
        # 筛选列
        df = df[self.feature_name]
        df = df.apply(pd.to_numeric)
        # 对原始数据进行可视化
        # data_visualize(df,self.feature_name[:-1])
        # 数据清洗，填充nan。连续变量用中位数，离散型变量根据相应的连续变量取值用0/1填充
        df['Number of sexual partners'] = df['Number of sexual partners'].fillna(
            df['Number of sexual partners'].median())
        df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
        df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
        # 离散变量与其对应的连续变量
        df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
        if df['Smokes (years)'].median() != 0:
            df['Smokes'] = df['Smokes'].fillna(1)
        else:
            df['Smokes'] = df['Smokes'].fillna(0)
        df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(
            df['Hormonal Contraceptives (years)'].median())
        if df['Hormonal Contraceptives (years)'].median() != 0:
            df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
        else:
            df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(0)
        df['IUD (years)'] = df['IUD (years)'].fillna(df['IUD (years)'].median())
        if df['IUD (years)'].median() != 0:
            df['IUD'] = df['IUD'].fillna(1)
        else:
            df['IUD'] = df['IUD'].fillna(0)
        df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
        if df['STDs (number)'].median() != 0:
            df['STDs'] = df['STDs'].fillna(1)
        else:
            df['STDs'] = df['STDs'].fillna(0)
        df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(
            df['STDs: Time since first diagnosis'].median())
        df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(
            df['STDs: Time since last diagnosis'].median())

        # 划分数据集，75%train, 25%test, 控制每次输出一样
        train_ds, test_ds = train_test_split(df, test_size=0.25, random_state=2021, shuffle=False)

        # 正则化，处理数据得到x, y
        scale = preprocessing.MinMaxScaler()
        # scale = preprocessing.StandardScaler()
        train_y = torch.tensor(train_ds['Biopsy'].values).float()
        train_X = torch.tensor(scale.fit_transform(train_ds.drop('Biopsy', axis=1))).float()
        test_y = torch.tensor(test_ds['Biopsy'].values).float()
        test_X = torch.tensor(scale.fit_transform(test_ds.drop('Biopsy', axis=1))).float()

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
