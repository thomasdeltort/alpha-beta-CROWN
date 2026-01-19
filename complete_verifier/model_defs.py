#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from typing import Optional, Sequence, Tuple, Type

import torch
from torch.nn import functional as F
import torch.nn as nn
from collections import OrderedDict
import math
import sys

########################################
# Defined the model architectures
########################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            # can do planes 32
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            # can only do planes 16, block1
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            # print("residual relu:", out.shape, out[0].view(-1).shape)
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        # print("residual relu:", out.shape, out[0].view(-1).shape)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_planes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_planes=in_planes)


class CResNet5(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet5, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out


class CResNet7(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(CResNet7, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        # print("conv1 relu", out.shape, out[0].view(-1).shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            # print("avg", out.shape)
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = self.linear(out)
            # print("output", out.shape)
        elif self.last_layer == "dense":
            out = out.view(out.size(0), -1)
            # print("view", out.shape)
            out = F.relu(self.linear1(out))
            # print("linear1 relu", out.shape, out[0].view(-1).shape)
            out = self.linear2(out)
            # print("output", out.shape)
        return out


def resnet4b():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")

def resnet2b():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet5_16_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="dense")

def cresnet5_16_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=True, last_layer="avg")


def cresnet5_8_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet5_8_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet5_4_dense_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet5_4_avg_bn():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")


def cresnet7_8_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="dense")

def cresnet7_8_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=True, last_layer="avg")


def cresnet7_4_dense_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="dense")

def cresnet7_4_avg_bn():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=True, last_layer="avg")


def cresnet5_16_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="dense")


def cresnet5_16_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=16, bn=False, last_layer="avg")


def cresnet5_8_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet5_8_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet5_4_dense():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet5_4_avg():
    return CResNet5(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


def cresnet7_8_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")

def cresnet7_8_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="avg")


def cresnet7_4_dense():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="dense")

def cresnet7_4_avg():
    return CResNet7(BasicBlock, num_blocks=2, in_planes=4, bn=False, last_layer="avg")


class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x, W in zip(xs, self.Ws) if W is not None)
        return out


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]


def model_resnet(in_ch=3, in_dim=32, width=1, mult=16, N=1):
    def block(in_filters, out_filters, k, downsample):
        if not downsample:
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else:
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)),
            nn.ReLU(),
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0),
                  None,
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)),
            nn.ReLU()
        ]

    conv1 = [nn.Conv2d(in_ch, mult, 3, stride=1, padding=3 if in_dim == 28 else 1), nn.ReLU()]
    conv2 = block(mult, mult * width, 3, False)
    for _ in range(N):
        conv2.extend(block(mult * width, mult * width, 3, False))
    conv3 = block(mult * width, mult * 2 * width, 3, True)
    for _ in range(N - 1):
        conv3.extend(block(mult * 2 * width, mult * 2 * width, 3, False))
    conv4 = block(mult * 2 * width, mult * 4 * width, 3, True)
    for _ in range(N - 1):
        conv4.extend(block(mult * 4 * width, mult * 4 * width, 3, False))
    layers = (
            conv1 +
            conv2 +
            conv3 +
            conv4 +
            [nn.Flatten(),
             nn.Linear(mult * 4 * width * 8 * 8, 1000),
             nn.ReLU(),
             nn.Linear(1000, 10)]
    )
    model = DenseSequential(
        *layers
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
    return model

# def test_base():
#     # the second label is dumb, always 0
#     model = nn.Sequential(
#         nn.Linear(2, 2),
#         nn.ReLU(),
#         nn.Linear(2, 2)
#     )
#     # import pdb; pdb.set_trace()
#     return model

def mnist_tiny_mlp():
    """A very small model for testing completeness."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

def mnist_fc():
    # cifar base
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    )
    return model


def cifar_model_base():
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    # cifar deep
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cnn_4layer():
    # cifar_cnn_a
    return cifar_model_wide()


def cnn_4layer_adv():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_adv4():
    # cifar_cnn_a_adv
    return cifar_model_wide()

def cnn_4layer_mix4():
    # cifar_cnn_a_mix4
    return cifar_model_wide()


def cnn_4layer_b():
    # cifar_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def cnn_4layer_b4():
    # cifar_cnn_b4
    return cnn_4layer_b()

def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

def cifar_conv_small():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv_small_sigmoid():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.Sigmoid(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.Sigmoid(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.Sigmoid(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv_big():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_marabou_small():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2,),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def cifar_marabou_medium():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2,),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def cifar_marabou_large():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2,),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2304, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def mnist_conv_small():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*5*5,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_conv_big():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_2_200():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    return model


def mnist_6_100():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100, 10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_9_100():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_6_200():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model

def mnist_9_200():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,200),
        nn.ReLU(),
        nn.Linear(200,10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model


def mnist_fc1():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    return model


def mnist_fc2():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fc3():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def mnist_fc_3_512():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_4_512():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

def mnist_fc_5_512():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_6_512():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_fc_7_512():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def mnist_madry_secret():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    return model


def cifar_conv1():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_conv3():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_conv4():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv5():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def cifar_conv6():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model


def MadryCNN():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_one_maxpool():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_no_maxpool():
    return nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
    )


def MadryCNN_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4, 8, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


def MadryCNN_one_maxpool_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


def MadryCNN_no_maxpool_tiny():
    return nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*7*7,128),
            nn.ReLU(),
            nn.Linear(128, 10)
    )


class TradesCNN(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class TradesCNN_one_maxpool(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, stride=2)),
            ('relu2', activ),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class TradesCNN_no_maxpool(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3, stride=2)),
            ('relu2', activ),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3, stride=2)),
            ('relu4', activ),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

############### Models from CROWN-IBP paper (Zhang et al. 2020) ###################

def crown_ibp_model_a_b(in_ch=3, in_dim=32, width=2, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def crown_ibp_model_c_d_e_f(in_ch=3, in_dim=32, kernel_size=3, width=2, linear_size=64):
    if linear_size is None:
        linear_size = width * 64
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*width*h*h, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def crown_ibp_model_g_h_i_j(in_ch=3, in_dim=32, width=1, linear_size=256):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


def crown_ibp_dm_large(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


############### Models from auto_LiRPA paper (Xu et al. 2020) ###################

def crown_ibp_dm_large_bn(in_ch=3, in_dim=32, width=64, linear_size=512):
    """The same as the DM-large model but with batch normalization layers."""
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

############# Models from IBP with short warmup (Shi et al. 2021) ####################

def crown_ibp_dm_large_bn_full(in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,num_class)
    )
    return model


class BasicBlock_eth(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        bn: bool = True,
        kernel: int = 3,
        in_dim: int = -1,
    ) -> None:
        super(BasicBlock_eth, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel

        kernel_size = kernel
        assert kernel_size in [1, 2, 3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b = []
        layers_b.append(
            nn.Conv2d(
                in_planes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=p_1,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (in_planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=stride,
            padding=p_1,
        )

        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(
            nn.Conv2d(
                planes,
                self.expansion * planes,
                kernel_size=kernel_size,
                stride=1,
                padding=p_2,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=1,
            padding=p_2,
        )
        if bn:
            layers_b.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_b = nn.Sequential(*layers_b)

        layers_a = [torch.nn.Identity()]
        if stride != 1 or in_planes != self.expansion * planes:
            layers_a.append(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=(not bn),
                )
            )
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_a = nn.Sequential(*layers_a)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.path_a(x) + self.path_b(x)
        return out

    def _getShapeConv(
        self,
        in_shape: Tuple[int, int, int],
        conv_shape: Tuple[int, ...],
        stride: int = 1,
        padding: int = 0,
    ) -> Tuple[int, int, int]:
        inChan, inH, inW = in_shape
        outChan, kH, kW = conv_shape[:3]

        outH = 1 + int((2 * padding + inH - kH) / stride)
        outW = 1 + int((2 * padding + inW - kW) / stride)
        return (outChan, outH, outW)


def getShapeConv(
    in_shape: Tuple[int, int, int],
    conv_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int]:
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class ResNet_eth(nn.Sequential):
    def __init__(
        self,
        block: Type[BasicBlock_eth],
        in_ch: int = 3,
        num_stages: int = 1,
        num_blocks: int = 2,
        num_classes: int = 10,
        in_planes: int = 64,
        bn: bool = True,
        last_layer: str = "avg",
        in_dim: int = 32,
        stride: Optional[Sequence[int]] = None,
    ):
        layers = []
        self.in_planes = in_planes
        if stride is None:
            stride = (num_stages + 1) * [2]

        layers.append(
            nn.Conv2d(
                in_ch,
                self.in_planes,
                kernel_size=3,
                stride=stride[0],
                padding=1,
                bias=not bn,
            )
        )

        _, _, in_dim = getShapeConv(
            (in_ch, in_dim, in_dim), (self.in_planes, 3, 3), stride=stride[0], padding=1
        )

        if bn:
            layers.append(nn.BatchNorm2d(self.in_planes))

        layers.append(nn.ReLU())

        for s in stride[1:]:
            block_layers, in_dim = self._make_layer(
                block,
                self.in_planes * 2,
                num_blocks,
                stride=s,
                bn=bn,
                kernel=3,
                in_dim=in_dim,
            )
            layers.append(block_layers)

        if last_layer == "avg":
            layers.append(nn.AvgPool2d(4))
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(
                    self.in_planes * (in_dim // 4) ** 2 * block.expansion, num_classes
                )
            )
        elif last_layer == "dense":
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(self.in_planes * block.expansion * in_dim ** 2, 100)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Linear(100, num_classes))
        else:
            exit("last_layer type not supported!")

        super(ResNet_eth, self).__init__(*layers)

    def _make_layer(
        self,
        block: Type[BasicBlock_eth],
        planes: int,
        num_layers: int,
        stride: int,
        bn: bool,
        kernel: int,
        in_dim: int,
    ) -> Tuple[nn.Sequential, int]:
        strides = [stride] + [1] * (num_layers - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, bn, kernel, in_dim=in_dim)
            )
            in_dim = layers[-1].out_dim
            layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), in_dim


def resnet2b_eth(bn: bool = False) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth, num_stages=1, num_blocks=2, in_planes=8, bn=bn, last_layer="dense"
    )


def resnet2b2_eth(bn: bool = True, in_ch: int = 3, in_dim: int = 32) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=in_ch,
        num_stages=2,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2],
    )


def resnet4b1(bn: bool = True) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[1, 1, 2, 2, 2],
    )


def resnet4b2(bn: bool = True) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=3,
        num_stages=4,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 1, 1],
    )


def resnet3b2(bn: bool = True) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=3,
        num_stages=3,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 2],
    )


def resnet3b2_no_bn(bn: bool = False) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=3,
        num_stages=3,
        num_blocks=1,
        in_planes=16,
        bn=bn,
        last_layer="dense",
        stride=[2, 2, 2, 2],
    )


def resnet9b(bn: bool = True) -> nn.Sequential:
    return ResNet_eth(
        BasicBlock_eth,
        in_ch=3,
        num_stages=3,
        num_blocks=3,
        in_planes=16,
        bn=bn,
        last_layer="dense",
    )


def mnist_conv_super() -> nn.Sequential:
    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64 * 18 * 18, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        ]
    )


class Step_carvana(nn.Module):
    def __init__(self, ori_carvana, gt):
        super(Step_carvana, self).__init__()
        self.ori_carvana = ori_carvana
        gt = torch.tensor(gt, dtype=torch.get_default_dtype()).reshape(1, 31, 47)  # 0 means 0-dim > 1-dim, 1 means 0-dim < 1-dim
        gt[gt == 1] = -1  # flip results when ground truth selecting 1-dim
        gt[gt == 0] = +1  # keep results when ground truth selecting 0-dim
        gt = gt.repeat(2, 1, 1).unsqueeze(0)  # reshape to NCHW.
        self.gt = torch.nn.Parameter(gt, requires_grad=False)
        self.step_value_zero = torch.nn.Parameter(torch.tensor(0., dtype=torch.get_default_dtype()), requires_grad=False)

    def forward(self, x):
        x = self.ori_carvana(x)
        x = x * self.gt  # flip x by ground truth label
        x = x[:, :1] - x[:, 1:]
        x = torch.heaviside(x, self.step_value_zero)
        x = x.flatten(1)
        x = x.sum(1, keepdim=True)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from deel import torchlip
import numpy as np 
import copy
from torch.nn.utils.parametrize import is_parametrized

def vanilla_export(model1):
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    dict_modified_layers = {}
    for (n1,p1), (n2,p2) in zip(model1.named_modules(), model2.named_modules()):
        #print(n1,type(p1), type(p2))
        assert n1 == n2
        if isinstance(p1, torch.nn.Conv2d) and is_parametrized(p1):
            new_conv = torch.nn.Conv2d(p1.in_channels, p1.out_channels, kernel_size=p1.kernel_size, stride=p1.stride, padding=p1.padding, padding_mode=p1.padding_mode,bias=(p1.bias is not None))
            new_conv.weight.data = p1.weight.data.clone()
            new_conv.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_conv
            #print("modified",n2,type(p1), type(new_conv),p1.in_channels, p1.out_channels, p1.kernel_size[0], p1.stride[0], p1.padding[0], p1.padding_mode,(p1.bias is not None))
            #setattr(model2, n2, new_conv)
            #print(n1,type(p1), type(getattr(model2, n2)))
        if isinstance(p1, torch.nn.Linear) and is_parametrized(p1):
            new_lin = torch.nn.Linear(p1.in_features, p1.out_features, bias=(p1.bias is not None))
            new_lin.weight.data = p1.weight.data.clone()
            new_lin.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            #setattr(model2, n2, new_lin)
            dict_modified_layers[n2] = new_lin
            #print("modified",n2,type(p1), type(new_lin))
            #print(n1,type(p1), type(getattr(model2, n2)))
    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        # print("modified",n2, type(getattr(lay, split_hierarchy[-1])),type(new_layer))
        setattr(lay, split_hierarchy[-1], new_layer)
    return model2

class GroupSort_General(nn.Module):
    """
    Applies GroupSort specifically on the channel dimension.
    
    It permutes the input from (N, C, ...) to (N, ..., C), applies the 
    sort logic so that pairs (c_2k, c_2k+1) are sorted, and then restores
    the original layout.
    """
    def __init__(self, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Permute to Channel Last
        # We assume the channel is at self.axis (usually 1).
        # We move that axis to the very end (-1).
        dims = list(range(x.dim()))
        # Remove the channel axis from its current spot and append to end
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        # permute returns a view, but we usually need contiguous memory for reshaping
        x_permuted = x.permute(dims).contiguous()
        
        # Capture the shape after permutation: (N, D1, D2, ..., C)
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        num_channels = permuted_shape[-1]
        
        if num_channels % 2 != 0:
             raise ValueError(
                f"The number of channels must be even, but got {num_channels} "
                f"for input shape {x.shape}."
            )

        # 2. Flatten for the sorting logic
        # We flatten everything except batch. Since Channel is now last,
        # adjacent elements in this flattened view correspond to adjacent channels.
        x_flat = x_permuted.reshape(batch_size, -1)

        # --- Sort Logic (Verifiable / Auto_Lirpa compatible) ---
        
        # Group into pairs. 
        # Because we are Channel Last, the last dim is C.
        # This reshaping groups (c0, c1), (c2, c3), etc.
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        # Calculate diff and apply ReLU to determine min/max without conditional branching
        # min(a,b) = x2 - ReLU(x2 - x1)
        # max(a,b) = x1 + ReLU(x2 - x1)
        diff = x2s + (-1*x1s)
        relu_diff = self.relu(diff)
        
        y1 = x2s + (-1*relu_diff) # The smaller value
        y2 = x1s + relu_diff # The larger value
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        
        # --- End Logic ---

        # 3. Restore Shape
        
        # First reshape back to the permuted shape (N, ..., C)
        output_permuted = sorted_flat.reshape(permuted_shape)
        
        # Finally, permute back to Channel First (N, C, ...)
        # We need to calculate the inverse permutation indices
        inv_dims = list(range(x.dim()))
        # Move the last dim (which is now channels) back to self.axis
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        
        output = output_permuted.permute(inv_dims)
        
        return output
    
    

class GroupSort2Optimized(nn.Module):
    # THIS IMPLEMENTATION IS NOT VERIFIABLE WITH auto_LiRPA
    # due to torch.max(a, b)
    def __init__(self):
        super(GroupSort2Optimized, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_reshaped = x.view(N, C // 2, 2, H, W)
        x1s = x_reshaped[:, :, 0, :, :]
        x2s = x_reshaped[:, :, 1, :, :]
        # Max + Sum Preservation Logic
        y2_max = torch.max(x1s, x2s)
        y1_min = (x1s + x2s) - y2_max
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        return sorted_pairs.view(N, C, H, W)


def MNIST_MLP():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
        # GroupSort_General(),

		nn.Linear(100, 100),
		nn.ReLU(),
        # GroupSort_General(),

		nn.Linear(100, 10)
	)
	return model

class MNIST_ConvSmall(nn.Module):
    def __init__(self):
        super(MNIST_ConvSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x_p = torch.relu(self.fc1(x))
        x = self.fc2(x_p)
        return x

class MNIST_ConvLarge(nn.Module):
    def __init__(self):
        super(MNIST_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def CIFAR10_CNN_A():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_CNN_B():
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )

def CIFAR10_CNN_C():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def CIFAR10_ConvSmall():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_ConvDeep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# class CIFAR10_ConvLarge(nn.Module):
#     def __init__(self):
#         super(CIFAR10_ConvLarge, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=512)
#         self.fc3 = nn.Linear(in_features=512, out_features=10)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = torch.relu(self.conv4(x))
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
def CIFAR10_ConvLarge():
    """
    Creates the CIFAR10_ConvLarge model using nn.Sequential.
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        
        nn.Flatten(),
        
        nn.Linear(in_features=64*8*8, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=10)
    )
    return model

    
# Note: This python file present every model we use for benchmarking. 
# These models correspond to the lipschitz version of all the model architectures from Wang et al. (2021) & Leino et al. (2021)
# This version uses ReLU as the activation function.

def MLP_MNIST_1_LIP():
    """
    Model: MLP_1_LIP (MNIST)
    Structure: Linear(784, 100) -> ReLU -> Linear(100, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        nn.ReLU(),
        torchlip.SpectralLinear(100, 100),
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvSmall_MNIST_1_LIP():
    """
    Model: ConvSmall_1_LIP (MNIST)
    Structure: Conv(1, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(1568, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_MNIST_1_LIP():
    """
    Model: ConvLarge_1_LIP (MNIST)
    Structure: Conv(1, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
                Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(3136, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def CNNA_CIFAR10_1_LIP():
    """
    Model: CNN-A_1_LIP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        # GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        # GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        nn.ReLU(),
        # GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def CNNB_CIFAR10_1_LIP():
    """
    Model: CNN-B_1_LIP (CIFAR-10)
    Structure: Conv(3, 32, 5, 2, 0) -> ReLU -> Conv(32, 128, 4, 2, 1) -> ReLU -> Linear(6272, 250) -> ReLU -> Linear(250, 10)
    Note: The paper specifies Linear(8192, 250), which implies an 8x8 feature map before flattening (128*8*8=8192).
          However, the specified convolutional layers produce a 7x7 feature map (128*7*7=6272).
          This implementation follows the specified conv layers, resulting in 6272 input features.
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        nn.ReLU(),
        torchlip.SpectralLinear(250, 10)
    )
    return vanilla_export(model)

def CNNC_CIFAR10_1_LIP():
    """
    Model: CNN-C_1_LIP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 0) -> ReLU -> Conv(8, 16, 4, 2, 0) -> ReLU -> Linear(576, 128) -> ReLU ->
                Linear(128, 64) -> ReLU -> Linear(64, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        nn.ReLU(),
        torchlip.SpectralLinear(128, 64),
        nn.ReLU(),
        torchlip.SpectralLinear(64, 10)
    )
    return vanilla_export(model)

def ConvSmall_CIFAR10_1_LIP():
    """
    Model: ConvSmall_1_LIP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 0) -> ReLU -> Conv(16, 32, 4, 2, 0) -> ReLU -> Linear(1152, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvDeep_CIFAR10_1_LIP():
    """
    Model: ConvDeep_1_LIP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU ->
                Conv(8, 8, 4, 2, 1) -> ReLU -> Linear(512, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_CIFAR10_1_LIP():
    """
    Model: ConvLarge_1_LIP (CIFAR-10)
    Structure: Conv(3, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
                Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(4096, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)


# Note: This python file present every model we use for benchmarking. 
# These models correspond to the lipschitz Gradient Norm Preserving version of all the model architectures from Wang et al. (2021) & Leino et al. (2021)
# This version uses Group Sort 2 as the activation function.

from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

def MLP_MNIST_1_LIP_GNP():
    """
    Model: MLP_1_LIP_GNP (MNIST)
    Structure: Linear(784, 100) -> ReLU -> Linear(100, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvSmall_MNIST_1_LIP_GNP():
    """
    Model: ConvSmall_1_LIP_GNP (MNIST)
    Structure: Conv(1, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(1568, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        #FIXME Convert to zero padding
        AdaptiveOrthoConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_MNIST_1_LIP_GNP():
    """
    Model: ConvLarge_1_LIP_GNP (MNIST)
    Structure: Conv(1, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
                Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(3136, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)


def CNNA_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-A_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def CNNB_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-B_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 32, 5, 2, 0) -> ReLU -> Conv(32, 128, 4, 2, 1) -> ReLU -> Linear(6272, 250) -> ReLU -> Linear(250, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        GroupSort_General(),
        torchlip.SpectralLinear(250, 10)
    )
    return vanilla_export(model)

def CNNC_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-C_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 0) -> ReLU -> Conv(8, 16, 4, 2, 0) -> ReLU -> Linear(576, 128) -> ReLU ->
                Linear(128, 64) -> ReLU -> Linear(64, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        GroupSort_General(),
        torchlip.SpectralLinear(128, 64),
        GroupSort_General(),
        torchlip.SpectralLinear(64, 10)
    )
    return vanilla_export(model)

def ConvSmall_CIFAR10_1_LIP_GNP():
    """
    Model: ConvSmall_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 0) -> ReLU -> Conv(16, 32, 4, 2, 0) -> ReLU -> Linear(1152, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvDeep_CIFAR10_1_LIP_GNP():
    """
    Model: ConvDeep_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU ->
                Conv(8, 8, 4, 2, 1) -> ReLU -> Linear(512, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_CIFAR10_1_LIP_GNP():
    """
    Model: ConvLarge_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
                Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(4096, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def VGG13_1_LIP_GNP_CIFAR10():
    """
    Model: VGG13-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
                [Conv(256) x 2] -> StridedConv -> [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 256 * 4 * 4 = 4096
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)


def VGG16_1_LIP_GNP_CIFAR10():
    """
    Model: VGG16-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
                [Conv(256) x 3] -> StridedConv -> [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 256 * 4 * 4 = 4096
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def VGG19_1_LIP_GNP_CIFAR10():
    """
    Model: VGG19-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
                [Conv(256) x 4] -> StridedConv -> [Conv(512) x 4] -> StridedConv ->
                [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1,padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 4: 256x4x4 -> 512x4x4
        AdaptiveOrthoConv2d(256, 512, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 512x4x4 -> 512x2x2
        AdaptiveOrthoConv2d(512, 512, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 512 * 2 * 2 = 2048
        torchlip.SpectralLinear(512 * 2 * 2, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

# Note: This section defines the 1_LIP_Bjork family.
# Characteristics: SpectralConv2d (default Bjorck), SpectralLinear, GroupSort 2 activation.

def MLP_MNIST_1_LIP_Bjork():
    """
    Model: MLP_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvSmall_MNIST_1_LIP_Bjork():
    """
    Model: ConvSmall_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_MNIST_1_LIP_Bjork():
    """
    Model: ConvLarge_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def CNNA_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-A_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def CNNB_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-B_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        GroupSort_General(),
        torchlip.SpectralLinear(250, 10)
    )
    return vanilla_export(model)

def CNNC_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-C_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        GroupSort_General(),
        torchlip.SpectralLinear(128, 64),
        GroupSort_General(),
        torchlip.SpectralLinear(64, 10)
    )
    return vanilla_export(model)

def ConvSmall_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvSmall_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvDeep_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvDeep_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return vanilla_export(model)

def ConvLarge_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvLarge_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def VGG13_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG13-like_1_LIP_Bjork (CIFAR-10)
    Structure matched to GNP version but with SpectralConv2d
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def VGG16_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG16-like_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)

def VGG19_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG19-like_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Block 4
        torchlip.SpectralConv2d(256, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(512, 512, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(512 * 2 * 2, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return vanilla_export(model)