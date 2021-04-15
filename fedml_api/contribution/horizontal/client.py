import logging


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        """客户端id号、该客户端的本地训练数据、本地测试数据、样本数目、参数、设备、模型训练器"""
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    # 更新该客户端，修改为编号client_idx的客户端情况，修改本地训练集、测试集、样本数目
    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    # 使用给定模型训练器为全局参数w在本地设备上通过本地训练数据进行训练
    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    # 在本设备上进行本地测试（评估），使用测试集或训练集
    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    # 获取训练数据的预测值
    def predict_on_test(self):
        metrics = self.model_trainer.predict(self.local_test_data, self.device)
        return metrics

    def show(self):
        return self.model_trainer.show(self.local_test_data)