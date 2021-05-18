import copy
import logging
import random

import numpy as np
import pandas as pd
import shap
import torch
import wandb

from fedml_api.contribution.horizontal.client import Client
from ..vertical.federate_shap import FederateShap


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, feature_name] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.feature_name = feature_name

        self.client_list = []  # 每轮的参与客户端
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    # 启动某轮次的客户端，初始化为客户端的前client_num_per_round名，之后每轮次用采样的客户端数据进行信息更新，得到该轮次的参与客户端
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        # 补充：判断总客户端数量是否小于每轮参与的客户端数量
        if self.args.client_num_in_total < self.args.client_num_per_round:
            logging.info("client_num_in_total is less than client_num_per_round, please check params")
        # 根据每轮参与的客户端数量生成相应个数的客户端(每轮参与的基本客户端列表)
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    # 获得全局参数（全局模型），在每个轮次中对参与的客户端采样，进行训练
    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            # 记录该轮次每个客户端所训练出的模型本地参数w
            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            # 采样选择本轮参与优化的客户端id索引
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            # logging.info("client_indexes = " + str(client_indexes))  # 输出内容与_client_sampling中的输出重复

            # 遍历每轮次的基本客户端列表，将对应序号采样的客户端信息填充其中并进行训练
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]  # 根据序号去获取对应采样的客户端id，并将相关数据更新到client_list中的该序号客户端上（填充信息）
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(w_global)
                # self.logger.info("local weights = " + str(w))
                logging.info(
                    "client_idx: {}, iteration: {}-th, local weights have been trained. ".format(client_idx, round_idx))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)
            logging.info("iteration: {}-th, global weights have been aggregated. ".format(round_idx))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    # 对某轮次中参与的客户端进行采样，选择该轮次中参与优化的客户端集
    # 补充：删除的客户端编号，不属于客户端集合【贡献量计算】
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round, delete_client=None):
        # 如果每轮参与通信的客户端数目等于全部客户端数目，则不用采样、全部选上参与优化
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        # 否则选择两者的最小数目作为采样数目（防止设置值过大情况），从总的客户端中抽样一定数目客户端选择本轮参与通信
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            # 考虑是否有删除客户端
            if delete_client is None:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            else:
                l1 = [x for x in range(client_num_in_total)]
                l1.pop(delete_client)
                client_indexes = np.random.choice(l1, num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # 从全局测试数据中生成验证集
    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)  # 根据采样索引获取采样的测试数据集
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)  # 将采样的测试数据分批次
        self.val_global = sample_testset

    # 聚合各个客户端的更新
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]  # 一直覆盖，可省略
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        # 对每个参数k，从所有本地客户端参数中按比聚合（权重由客户端样本数在总样本数中的占比决定）
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    # 对所有客户端进行本地测试
    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        # 基准参与客户端，用来更新数据、进行处理
        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # 如果该客户端没有本地测试数据映射，则跳过
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)  # 原：false
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    # 在验证集上进行本地测试【此处只针对StackOverflow数据测试】
    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        # 若不存在验证集则从全局训练数据中生成
        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)

    # 删除某客户端子集后的模型训练
    def train_with_delete(self, delete_client):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            # 记录该轮次每个客户端所训练出的模型本地参数w
            w_locals = []

            # 采样选择本轮参与优化的客户端id索引
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round, delete_client)

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(w_global)
                # self.logger.info("local weights = " + str(w))
                logging.info(
                    "client_idx: {}, iteration: {}-th, local weights have been trained. ".format(client_idx, round_idx))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round, return Y
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    # 在测试数据集上进行预测
    def predict_on_test(self):
        logging.info("################predict on test data")

        test_metrics = {
            'y_true': [],
            'y_pred': [],
            'y_predicted': []
        }

        # 基准参与客户端，用来更新数据、进行处理
        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # 如果该客户端没有本地测试数据映射，则跳过
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # predict
            test_local_metrics = client.predict_on_test()
            test_metrics['y_true'].append(copy.deepcopy(test_local_metrics['y_true']))
            test_metrics['y_pred'].append(copy.deepcopy(test_local_metrics['y_pred']))
            test_metrics['y_predicted'].append(copy.deepcopy(test_local_metrics['y_predicted']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        return test_metrics

    # 在所有数据上进行计算特征的shapley值，并计算联邦特征的shapley值
    def show_shap_on_all(self):
        logging.info("################load data for shap")

        train_X_all, test_X_all = torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        client = self.client_list[0]
        for client_idx in range(self.args.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            train_X, train_y, test_X = client.get_all_X()
            train_X_all = torch.cat((train_X_all, train_X), 0)
            test_X_all = torch.cat((test_X_all, test_X), 0)

        # explain model [total]
        feature_num = len(self.feature_name) - 1  # 特征个数
        e = shap.DeepExplainer(self.model_trainer.model, train_X_all)  # train_X_all
        shap_values = e.shap_values(train_X_all)  # test_X_all
        # 条状图，各特征shapely的均值；蜂巢图，shapely值分别
        shap.summary_plot(shap_values[1], train_X_all, feature_names=self.feature_name[:-1],
                          max_display=feature_num, sort=False, plot_size=(12, 8))
        shap.summary_plot(shap_values[1], train_X_all, feature_names=self.feature_name[:-1], plot_type="bar",
                          max_display=feature_num, sort=False, plot_size=(12, 8))
        # 对单个例子的shap值进行显示
        shap.bar_plot(shap_values[0][99], feature_names=self.feature_name[:-1], max_display=feature_num)
        shap.bar_plot(shap_values[1][99], feature_names=self.feature_name[:-1], max_display=feature_num)
        # 力图：显示某样本为1类时各特征的驱动情况
        shap.force_plot(e.expected_value[1], np.around(shap_values[1][99], decimals=4), train_X_all[99].numpy(),
                        feature_names=self.feature_name[:-1], matplotlib=True, figsize=(25, 6))

        # mnist绘图
        # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        # test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        # shap.image_plot(shap_numpy, -test_numpy)

        # compute federate shapley
        train_X_all_pd = pd.DataFrame(train_X_all.numpy())
        data = shap.kmeans(train_X_all_pd, 20)
        weights = data.weights
        step = 3

        # 遍历联邦特征，获取到当前联邦特征的起始下标
        for fed_pos in range(0, feature_num, step):
            cols_federated = self.feature_name[:-1]
            cols_federated[fed_pos] = 'Federated'
            del cols_federated[fed_pos + 1: fed_pos + 3]
            # 修改shap_values的各项值（list）
            shap_values_fed = copy.deepcopy(shap_values)
            for i, val in enumerate(shap_values_fed):
                # 遍历当前联邦特征，计算联邦特征的shapley值
                sumFed = np.zeros(val.shape[0], dtype=float)
                sumWeights = 0
                for fed_pos_index in range(fed_pos, fed_pos+step):
                    sumFed += val[:, fed_pos_index] * weights[fed_pos_index]
                    sumWeights += weights[fed_pos_index]
                # 计算联邦特征的shapley值
                val[:, 0] = sumFed / sumWeights
                val = np.delete(val, [1, 2], 1)
                compareWeight = sumWeights / 3
                # 遍历其他各列，根据权重进行加减，对shapley值进行修正
                for l in range(feature_num - step + 1):
                    if l == fed_pos:
                        continue
                    else:
                        if weights[l] > compareWeight:
                            val[:, l] += sumFed / sumWeights * 10 * weights[l]
                        else:
                            val[:, l] -= sumFed / sumWeights * weights[l]
                shap_values_fed[i] = val
            # 对联邦特征的shapley值进行绘图
            shap.summary_plot(shap_values_fed[1], feature_names=cols_federated, sort=False, plot_size=(12, 8),
                              color='y')
            shap.summary_plot(shap_values_fed[1], feature_names=cols_federated, sort=False, plot_type="bar",
                              plot_size=(12, 8), color='y')

    def show_federate_shap_on_each_client(self):
        logging.info("################load data for shap")

        client = self.client_list[0]
        fed_pos = 0  # 联邦特征的起始下标
        for client_idx in range(self.args.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            train_X, train_y, test_X = client.get_all_X()

            # federate shapley
            train_X_all_pd = pd.DataFrame(train_X.numpy())
            f_knn = lambda x: self.model_trainer.model.forward(x)
            med = train_X_all_pd.median().values.reshape((1, train_X_all_pd.shape[1]))
            feature_num = len(self.feature_name) - 1  # 特征个数
            fs = FederateShap()

            # Aggregated and average federated shap
            data = shap.kmeans(train_X_all_pd, 20)
            step = 3
            shap_values_whole = []
            cols_federated = self.feature_name[:-1]
            cols_federated[fed_pos] = 'Federated'
            del cols_federated[fed_pos + 1: fed_pos + step]

            for x in data.data:
                phi = fs.kernel_shap_federated_with_step(f_knn, x, med, feature_num, fed_pos, step)
                base_value = phi[-1]
                shap_values = phi[:-1]
                shap_values_whole.append(list(shap_values))
            shap_values_whole = np.array(shap_values_whole)
            shap_values_whole_mean = np.mean(shap_values_whole, axis=0).transpose()
            # 绘制联邦特征shapley值的图
            shap.summary_plot(shap_values_whole_mean, feature_names=cols_federated, sort=False)
            shap.summary_plot(shap_values_whole_mean, feature_names=cols_federated, sort=False, plot_type="bar")
            fed_pos += step

    def test_federated_shap(self):
        ####### test shap federated
        import shap
        shap.initjs()

        # load data
        X, y = shap.datasets.adult()
        cols = ['Age', 'Country', 'Education-Num', 'Marital Status', 'Relationship', 'Race', 'Sex', 'Capital Gain',
                'Capital Loss', 'Workclass', 'Occupation', 'Hours per week']
        X = X[cols]
        # X_display, y_display = shap.datasets.adult(display=True)
        import sklearn
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2,
                                                                                      random_state=7)
        # normalize data
        dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
        X_train_norm = X_train.copy()
        X_valid_norm = X_valid.copy()
        for k, dtype in dtypes:
            m = X_train[k].mean()
            s = X_train[k].std()
            X_train_norm[k] -= m
            X_train_norm[k] /= s

            X_valid_norm[k] -= m
            X_valid_norm[k] /= s
        # train model
        knn_norm = sklearn.neighbors.KNeighborsClassifier()
        knn_norm.fit(X_train_norm, y_train)
        # test score
        knn_norm.score(X_valid, y_valid)

        # Explain the model
        f_knn = lambda x: knn_norm.predict_proba(x)[:, 1]
        med = X_train_norm.median().values.reshape((1, X_train_norm.shape[1]))
        x = np.array(X_train_norm.iloc[0])
        # x = np.array(X_train_norm.loc[2583])
        M = 12
        from ..vertical.federate_shap import FederateShap
        fs = FederateShap()

        # shap
        phi = fs.kernel_shap(f_knn, x, med, M)
        base_value = phi[-1]
        shap_values = phi[:-1]
        import pandas as pd
        shap_values_df = pd.DataFrame(data=np.array([shap_values]), columns=list(X_train_norm))
        print("Shap Values")
        # shap_values_df
        import matplotlib.pyplot as plt
        row = shap_values_df.iloc[0]
        row.plot(kind='bar', color='k')
        plt.show()

        # federated shap
        fed_pos = 9
        print(x)
        phi = fs.kernel_shap_federated(f_knn, x, med, M, fed_pos)
        base_value = phi[-1]
        shap_values = phi[:-1]
        new_columns = list(X_train_norm)[:fed_pos]
        new_columns.extend(['Federated'])
        shap_values_df = pd.DataFrame(data=np.array([shap_values]), columns=new_columns)
        print("Federated Shap Values")
        # shap_values_df.plot()
        row = shap_values_df.iloc[0]
        row.plot(kind='bar', color='b')
        plt.show()
