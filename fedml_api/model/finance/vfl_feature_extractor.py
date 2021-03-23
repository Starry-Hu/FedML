import torch.nn as nn


class VFLFeatureExtractor(nn.Module):
    """
    特征提取器：
    仅定义了神经网络和输出维度，没有相应的优化参数等（相比models_standalone中的模型）
    方法仅提供了前向传播和获取输出维度的方法
    """
    def __init__(self, input_dim, output_dim):
        super(VFLFeatureExtractor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.LeakyReLU()
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.classifier(x)

    def get_output_dim(self):
        return self.output_dim
