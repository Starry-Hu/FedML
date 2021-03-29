import copy
import logging

import torch
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel

class CentralizedTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    拿到所有客户端的数据后在云端集中进行参数优化得到模型，之后在这些客户端上进行性能评估
    """

    def __init__(self, dataset, model, device, args):
        self.device = device  # 一般指cpu/gpu
        self.args = args
        # 数据集各项装载：[训练数据大小，测试数据大小，训练数据批情况，测试数据批情况，各客户端训练数据大小映射，各客户端本地训练数据批情况，各客户端本地测试数据批情况，类别数]
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model = model
        self.model.to(self.device)  # model.to：将模型加载到指定设备上，常见的有cpu和cuda
        self.criterion = nn.CrossEntropyLoss()  # 设置标准：进行交叉熵计算
        # 利用pytorch对参数进行优化，根据客户端优化器参数选择相应的优化器
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:  # 默认使用Adam，注意所需的那些参数情况
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def train(self):
        for epoch in range(self.args.epochs):
            if self.args.data_parallel == 1:  # 判断是否并行，（针对单服务器多gpu数据并行，并行只存在于前向传播中）
                self.train_global.sampler.set_epoch(epoch)  # 对全局训练数据批情况设置迭代次数
            # 进行具体的训练，每迭代训练一次后，进行模型性能评估
            self.train_impl(epoch)
            self.eval_impl(epoch)


    # 某一次迭代的具体训练过程，使用批梯度下降
    def train_impl(self, epoch_idx):
        self.model.train()  # 此处调用的torch.nn.model中的训练
        # 处理全局训练数据的批情况，分批出进行数据训练（整体流程和pytorch cnn一致）
        for batch_idx, (x, labels) in enumerate(self.train_global):
            # logging.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)  # 1. 准备好进入模型的数据，转化为一致类型
            labels = labels.long()  # fix bug：转化为long类型，否则计算损失函数出错
            self.optimizer.zero_grad()  # 2. 积累梯度，进入实例之前将优化的参数梯度设置为0
            log_probs = self.model(x)  # 3. 得到输出
            loss = self.criterion(log_probs, labels)  # 4. 计算损失函数，此处使用交叉熵损失函数(criterion)
            loss.backward()  # 5. 反向传播并更新梯度
            self.optimizer.step()  # 6. 优化器进行一步优化
            logging.info('Local Training Epoch: {} {}-th iters\t Loss: {:.6f}'.format(epoch_idx,
                                                                                      batch_idx, loss.item()))

    # 每迭代一定次数进行一次客户端上的对测试集和训练集的性能评估
    def eval_impl(self, epoch_idx):
        # 源代码对于train和test的评估都是一样的，此处进行修改
        # train
        if epoch_idx % self.args.frequency_of_train_acc_report == 0:
            self.test_on_all_clients(b_is_train=True, epoch_idx=epoch_idx)

        # test
        # if epoch_idx % self.args.frequency_of_train_acc_report == 0:
        if epoch_idx % self.args.frequency_of_test_acc_report == 0:  # fix bug
            self.test_on_all_clients(b_is_train=False, epoch_idx=epoch_idx)

    # 对所有客户端进行性能评估
    def test_on_all_clients(self, b_is_train, epoch_idx):
        # 此时结果不参与模型优化
        self.model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        # b_is_train代表使用训练集或测试集评估
        if b_is_train:
            test_data = self.train_global
        else:
            test_data = self.test_global
        with torch.no_grad():  # 上下文管理器，被该语句包裹起来的部分将不会track梯度（不参与计算图的构建，不会被记录用于反向传播）
            for batch_idx, (x, target) in enumerate(test_data):
                # 封装数据，计算预测值，计算损失函数
                x = x.to(self.device)
                target = target.to(self.device)
                target = target.long()  # fix bug：转化为long类型，否则计算损失函数出错
                pred = self.model(x)
                loss = self.criterion(pred, target)

                # 对数据集是stackoverflow进行特定处理，为啥特定处理？
                if self.args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()  # 正确样本数
                metrics['test_loss'] += loss.item() * target.size(0)  # 损失值
                metrics['test_total'] += target.size(0)  # 总样本数
        # rank代表是否保存评估日志？
        if self.args.rank == 0:
            self.save_log(b_is_train=b_is_train, metrics=metrics, epoch_idx=epoch_idx)

    # 保存性能评估日志，最终输出到wandb上
    def save_log(self, b_is_train, metrics, epoch_idx):
        prefix = 'Train' if b_is_train else 'Test'

        # 记录各次评估的参数情况
        all_metrics = {
            'num_samples': [],
            'num_correct': [],
            'precisions': [],
            'recalls': [],
            'losses': []
        }

        all_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        all_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        all_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        if self.args.dataset == "stackoverflow_lr":
            all_metrics['precisions'].append(copy.deepcopy(metrics['test_precision']))
            all_metrics['recalls'].append(copy.deepcopy(metrics['test_recall']))

        # performance on all clients，计算客户端上的性能表现（百分比）
        acc = sum(all_metrics['num_correct']) / sum(all_metrics['num_samples'])
        loss = sum(all_metrics['losses']) / sum(all_metrics['num_samples'])
        precision = sum(all_metrics['precisions']) / sum(all_metrics['num_samples'])
        recall = sum(all_metrics['recalls']) / sum(all_metrics['num_samples'])

        # 写入结果
        if self.args.dataset == "stackoverflow_lr":
            stats = {prefix + '_acc': acc, prefix + '_precision': precision, prefix + '_recall': recall,
                     prefix + '_loss': loss}
            wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
            wandb.log({prefix + "/Pre": precision, "epoch": epoch_idx})
            wandb.log({prefix + "/Rec": recall, "epoch": epoch_idx})
            wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)
        else:
            stats = {prefix + '_acc': acc, prefix + '_loss': loss}
            wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
            wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)

        # stats = {prefix + '_acc': acc, prefix + '_loss': loss}
        # wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
        # wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
        # logging.info(stats)