#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : mlp.py

"""
An artificial neural network as a troubled-cell indicator
网络类型：MLP
网络结构：
    input layer's size: 5
    number of hidden layers: 5
    hidden layers' size: 256, 128, 64, 32, 16
    activation function of above layers: ReLU
    output layer's size: 2
    activation function of output layer: Softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(5, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.hidden_layer3 = nn.Linear(128, 64)
        self.hidden_layer4 = nn.Linear(64, 32)
        self.hidden_layer5 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=1)
        self.size = self.__size()

    def forward(self, x):
        out = F.relu(self.hidden_layer1(x))
        out = F.relu(self.hidden_layer2(out))
        out = F.relu(self.hidden_layer3(out))
        out = F.relu(self.hidden_layer4(out))
        out = F.relu(self.hidden_layer5(out))
        out = self.output_layer(out)
        out = self.softmax(out)
        return out

    def __size(self):
        """
        返回网络的超参数数量
        """
        num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num


if __name__ == "__main__":
    xx = torch.rand((3, 5))
    net = MLP()
    print(net.size)
    print(net(xx))
