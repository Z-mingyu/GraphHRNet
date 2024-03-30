# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch.nn as nn

from models.gconv.vanilla_graph_conv import DecoupleVanillaGraphConv
from models.gconv.pre_agg_graph_conv import DecouplePreAggGraphConv
from models.gconv.post_agg_graph_conv import DecouplePostAggGraphConv
from models.gconv.conv_style_graph_conv import ConvStyleGraphConv
from models.gconv.no_sharing_graph_conv import NoSharingGraphConv
from models.gconv.modulated_gcn_conv import ModulatedGraphConv
from models.gconv.sem_graph_conv import SemGraphConv

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, gcn_type=None):
        super(_GraphConv, self).__init__()

        if gcn_type == 'vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_vanilla':
            self.gconv = DecoupleVanillaGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_preagg':
            self.gconv = DecouplePreAggGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj, decouple=False)
        elif gcn_type == 'dc_postagg':
            self.gconv = DecouplePostAggGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'convst':
            self.gconv = ConvStyleGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'nosharing':
            self.gconv = NoSharingGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'modulated':
            self.gconv = ModulatedGraphConv(input_dim, output_dim, adj)
        elif gcn_type == 'semantic':
            self.gconv = SemGraphConv(input_dim, output_dim, adj)
        else:
            assert False, 'Invalid graph convolution type'

        self.bn = nn.BatchNorm1d(output_dim)

        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)  # 转置，根据后续代码来看应该是和论文中的计算方式有点区别
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, adj, input_dim, output_dim, p_dropout, gcn_type, nodes_group, pooling=False,
                 channel_change=False):

        super(BasicBlock, self).__init__()
        self.conv1 = _GraphConv(adj, input_dim, output_dim, p_dropout, gcn_type)

        self.conv2 = _GraphConv(adj, output_dim, output_dim, p_dropout, gcn_type)

        self.channel_change = channel_change
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.pooling:
            out = self.pool(out)
            residual = self.pool(x)

        if self.channel_change:
            residual = self.gconv3(residual)

        out = out + residual
        out = self.relu(out)  # 在定义的图卷积中加入了relu的，这里再用以此好像没必要了，不过residual可能需要

        return out


class Bottleneck(nn.Module):
    expansion = 2  # 鉴于特征融合大概是节点数除以2，所以特征长度乘以2

    def __init__(self, adj, input_dim, output_dim, p_dropout, gcn_type, channel_change=False):
        super(Bottleneck, self).__init__()
        self.gconv1 = _GraphConv(adj, input_dim, output_dim, p_dropout, gcn_type)

        self.gconv2 = _GraphConv(adj, input_dim, output_dim * self.expansion, p_dropout, gcn_type)

        self.gconv3 = _GraphConv(adj, output_dim, output_dim * self.expansion, p_dropout, gcn_type)
        self.channel_change = channel_change
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.gconv1(x)

        # out = self.gconv2(out)

        out = self.gconv3(out)

        if self.channel_change:
            residual = self.gconv2(residual)
        out = out + residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, adj, block, layers, p_dropout, gcn_type):
        self.inplanes = 64
        super(PoseResNet, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.gconv1 = _GraphConv(adj, 2, 64, p_dropout, gcn_type)
        self.bn1 = nn.BatchNorm1d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(adj, block, 64, layers[0], p_dropout, gcn_type)
        self.layer2 = self._make_layer(adj, block, 64, layers[1], p_dropout, gcn_type)
        self.layer3 = self._make_layer(adj, block, 64, layers[2], p_dropout, gcn_type)
        self.layer4 = self._make_layer(adj, block, 64, layers[3], p_dropout, gcn_type)

        self.gconv_output = nn.Conv1d(128, 3, (1,))

    def _make_layer(self, adj, block, planes, blocks, p_dropout, gcn_type):
        channel_change = None
        if self.inplanes != planes * block.expansion:
            channel_change = True

        layers = []
        layers.append(block(adj, self.inplanes, planes, p_dropout, gcn_type, channel_change))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(adj, self.inplanes, planes, p_dropout, gcn_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gconv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.Sigmoid(x.transpose(1, 2))
        out = self.gconv_output(x).transpose(1, 2)
        return out[:, :23, :], out[:, 23:91, :], out[:, 91:112, :], out[:, 112:, :]

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [1, 1, 1, 1]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(is_train, adj, p_dropout, gcn_type, num_layers, INIT_WEIGHTS=False):
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(adj, block_class, layers, p_dropout, gcn_type)

    if is_train and INIT_WEIGHTS:
        model.init_weights()

    return model
