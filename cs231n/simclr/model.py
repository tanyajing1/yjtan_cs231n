import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()
        # 构建编码器backbone，基于ResNet50
        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                # 修改第一个卷积层：将7x7卷积改为3x3，stride从2改为1
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        # 通过编码器提取特征
        x = self.f(x)
        # 展平特征图为向量
        feature = torch.flatten(x, start_dim=1)
        # 通过投影头得到对比学习特征
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
