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


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2), nn.Softmax(dim=1)
        )
        self.size = self.__size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out

    def __size(self) -> int:
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
