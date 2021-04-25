import torch
import matplotlib.pyplot as plt
import logging
import wandb


class DeleteMeasure(object):
    def __init__(self, client_num, base_metrics, metrics_list, device):
        self.client_num = client_num  # 客户端数目
        self.base_metrics = base_metrics  # 最初模型f的损失值
        self.metrics_list = metrics_list  # 删除某客户端后训练的模型f'损失值
        self.device = device

    # 计算客户端d_k影响力
    def compute_influence(self):
        influence_list = [None] * self.client_num
        for client in range(self.client_num):
            # 如果没有该客户端对应模型的预测值则跳过
            if self.metrics_list[client] is None:
                continue
            # 计算影响力（贡献）
            batch_sum = torch.tensor([], device=self.device)
            origin_pred = self.base_metrics["y_pred"]
            delete_pred = self.metrics_list[client]["y_pred"]
            # 遍历每个batch的预测结果
            for batch in range(len(origin_pred)):
                # 对应元素相减取绝对值，总的求和取平均（一维tensor形式）
                batch_one = torch.sum(torch.abs(origin_pred[batch] - delete_pred[batch])) / origin_pred[batch].size()[0]
                batch_sum = torch.cat((batch_sum, torch.tensor([batch_one])), 0)
            # 对每个batch的计算结果再求和取平均，等同于对所有实例相减取绝对值再求和取平均
            influence_list[client] = torch.sum(batch_sum) / batch_sum.size()[0]

            # predict on test dataset
            stats = {'client_idx': client, "influence": influence_list[client]}
            wandb.log({"influence": influence_list[client], 'client_idx': client})
            logging.info(stats)
        return influence_list

    def drawBarFigure(self, influence_list):
        client_axis = ['client-{}'.format(x) for x in range(self.client_num)]
        plt.bar(range(len(influence_list)), influence_list, width=0.8)
        plt.xticks(range(self.client_num), client_axis)
        plt.xlabel("Group(Clients) ID")
        plt.ylabel("Influence Instance Groups(Clients)")
        # plt.savefig("./bar1.png")
        plt.show()