import torch
from torch import nn
import shap

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                labels = labels.long()  # fix bug：转化为long类型，否则计算损失函数出错
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                target = target.long()  # fix bug：转化为long类型，否则计算损失函数出错
                pred = model(x)

                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    # 进行预测
    def predict(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        test_metrics = {
            'y_true':  torch.tensor([], dtype=torch.long, device=device),  # 数据标签
            'y_pred':  torch.tensor([], device=device),  # 模型全部输出
            'y_predicted': torch.tensor([], device=device)  # 模型最终输出的类别
        }

        # deactivate autograd engine and reduce memory usage and speed up computations
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                target = target.long()  # fix bug：转化为long类型，否则计算损失函数出错
                pred = model(x)  # 输出：每类的预测概率

                test_metrics['y_true'] = torch.cat((test_metrics['y_true'], target), 0)
                test_metrics['y_pred'] = torch.cat((test_metrics['y_pred'], pred), 0)

        _, test_metrics['y_predicted'] = torch.max(test_metrics['y_pred'], -1)  # 得到最终输出的类别
        return test_metrics

        # import torch.nn.functional as F  # 激励函数
        # with torch.no_grad():
        #     for data in test_loader:
        #         inputs = [i.to(device) for i in data[:-1]]
        #         labels = data[-1].to(device)
        #
        #         outputs = model(*inputs)
        #         y_true = torch.cat((y_true, labels), 0)
        #         all_outputs = torch.cat((all_outputs, outputs), 0)
        # _, y_pred = torch.max(all_outputs, 1)
        # y_pred_prob = F.softmax(all_outputs, dim=1)
        # return y_true, y_pred, y_pred_prob

    def show(self, test_loader):
        import numpy as np
        batch = next(iter(test_loader))
        images, _ = batch
        inds = np.random.choice(images.shape[0], 1, replace=False)

        background = images[0:1]
        test_images = images[1:]

        return background,test_images

        # next_x, _ = next(iter(test_loader))
        # print(next_x.shape)
        # np.random.seed(0)
        # inds = np.random.choice(next_x.shape[0], 1, replace=False)
        # e = shap.DeepExplainer(test_loader, next_x[inds, :])
        # test_x, _ = next(iter(test_loader))
        # shap_values = e.shap_values(test_x[:1])
        #
        # self.model.eval()
        # self.model.zero_grad()
        # with torch.no_grad():
        #     diff = (self.model(test_x[:1]) - self.model(next_x[inds, :])).detach().numpy().mean(0)
        # sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
        # d = np.abs(sums - diff).sum()
        # assert d / np.abs(diff).sum() < 0.001, "Sum of SHAP values does not match difference! %f" % (
        #             d / np.abs(diff).sum())