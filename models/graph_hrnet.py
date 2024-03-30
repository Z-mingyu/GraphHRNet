# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


import os
import logging

import torch
import torch.nn as nn

from common.graph_utils import adj_mx_from_edges

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

        self.pooling = pooling
        self.pool = _SkeletalPool(nodes_group)
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


class _SkeletalPool(nn.Module):
    def __init__(self, nodes_group):  # nodes_group是节点编号组合
        super(_SkeletalPool, self).__init__()
        self.high_group = sum(nodes_group, [])  # 多个列表合成一个
        self.mid_group = [0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10, 11, 12, 20, 13, 19, 14, 18, 15, 17, 16, 38, 21,
                          25, 22, 24, 23, 26, 27, 29, 28, 30, 31, 34, 32, 35, 33, 36, 37, 39, 40, 41, 42, 43, 44, 45,
                          46, 57, 47, 58, 48, 59, 49, 60, 50, 61, 51, 62, 52, 63, 53, 64, 54, 65, 55, 66, 56, 67]
        # 如果是whole_body，body+feet（23->12->6）,hand(21->10->5),face(68->34->17)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.shape[1] == 133:  # x.shape=[batch,node_nums,token_size]
            out = self.pool(x[:, self.high_group].transpose(1, 2))
            return out.transpose(1, 2)
        elif x.shape[1] == 68:
            out = self.pool(x[:, self.mid_group].transpose(1, 2))  # 第二级是严格按照顺序的，那么可以直接相邻的节点特征做pooling
            return out.transpose(1, 2)
        else:
            assert False, 'Invalid Type in Skeletal Pooling : x.shape is {}'.format(x.shape)


class _SkeletalUnpool(nn.Module):
    def __init__(self):
        super(_SkeletalUnpool, self).__init__()
        self.inv_low = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 11, 12, 13, 12, 11, 13, 14, 15,
                        14, 15, 16, 17, 18, 16, 17, 18, 19, 10, 19, 20, 20, 21, 21, 22, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

        self.inv_mid = [2, 0, 0, 1, 1, 3, 3, 5, 4, 5, 4, 7, 6, 7, 6, 9, 8, 11, 11, 9, 10, 10, 8, 12, 12, 13, 13, 14, 14,
                        15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27,
                        27, 28, 28, 29, 30, 30, 31, 32, 32, 31, 33, 33, 34, 35, 35, 34, 36, 36, 37, 37, 38, 29, 38, 39,
                        39, 40, 40, 16, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51,
                        51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63,
                        64, 64, 65, 65, 66, 66, 67, 67]  # self.inv_mid[80]=16,self.inv_mid[74]=29直接给出索引对应关系得到上采样的结果

    def forward(self, x):
        if x.shape[1] == 68:
            return x[:, self.inv_mid]
        elif x.shape[1] == 34:
            return x[:, self.inv_low]
        else:
            assert False, 'Invalid Type in Skeletal Unpooling : x.shape is {}'.format(x.shape)


class Bottleneck(nn.Module):
    expansion = 2  # 鉴于特征融合大概是节点数除以2，所以特征长度乘以2

    def __init__(self, adj, input_dim, output_dim, p_dropout, gcn_type, nodes_group, pooling=False,
                 channel_change=False):
        super(Bottleneck, self).__init__()
        self.gconv1 = _GraphConv(adj, input_dim, output_dim, p_dropout, gcn_type)

        self.gconv2 = _GraphConv(adj, output_dim, output_dim, p_dropout, gcn_type)

        self.gconv3 = _GraphConv(adj, output_dim, output_dim * self.expansion, p_dropout, gcn_type)
        self.pooling = pooling
        self.pool = _SkeletalPool(nodes_group)
        self.channel_change = channel_change
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.gconv1(x)

        out = self.gconv2(out)

        out = self.gconv3(out)

        if self.pooling:
            out = self.pool(out)
            residual = self.pool(x)

        if self.channel_change:
            residual = self.gconv3(residual)
        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, adj, nodes, p_dropout, gcn_type, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.relu = nn.ReLU(True)
        self.nodes = nodes
        self.adj = adj
        self.p_dropout = p_dropout
        self.gcn_type = gcn_type
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        pooling = False
        channel_change = False

        if stride != 1:
            pooling = True
        if self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            channel_change = True

        layers = [block(
            self.adj[branch_index],
            self.num_inchannels[branch_index],
            num_channels[branch_index],
            self.p_dropout,
            self.gcn_type,
            self.nodes,
            pooling,
            channel_change
        )]
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.adj[branch_index],
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    self.p_dropout,
                    self.gcn_type,
                    self.nodes,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)  # modulelist可以自定义forward函数，而非像sequential只能顺序执行

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        #
        for i in range(num_branches if self.multi_scale_output else 1):
            # 最匪夷所思的一点是只有在最后一层是输出最大的特征图，但是前面说的是最后一层输出多尺寸特征图，反过来了
            fuse_layer = []
            for j in range(num_branches):  # 只取相邻的层来做融合，避免反复上下采样
                if j > i:
                    if abs(i - j) == 1:
                        fuse_layer.append(
                            nn.Sequential(
                                _GraphConv(self.adj[j],
                                           num_inchannels[j],
                                           num_inchannels[i],
                                           self.p_dropout,
                                           self.gcn_type
                                           ),

                                _SkeletalUnpool()
                            )
                        )
                    else:
                        fuse_layer.append(None)
                elif j == i:
                    fuse_layer.append(None)
                else:
                    if abs(i - j) == 1:

                        fuse_layer.append(
                            nn.Sequential(
                                _GraphConv(self.adj[j],
                                           num_inchannels[j],
                                           num_inchannels[i],
                                           self.p_dropout,
                                           self.gcn_type
                                           ),

                                _SkeletalPool(self.nodes)
                            )
                        )

                    else:
                        fuse_layer.append(None)

            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = torch.zeros_like(x[i])
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif abs(i - j) == 1:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class SEBlock(nn.Module):
    def __init__(self, adj, input_dim, reduction_ratio=8):
        super(SEBlock, self).__init__()
        hid_dim = input_dim // reduction_ratio
        self.fc1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.fc2 = nn.Linear(hid_dim, input_dim, bias=True)
        self.gap = nn.AvgPool1d(kernel_size=adj.shape[-1])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.gap(x)
        out = self.relu(self.fc1(out.squeeze()))
        out = self.sigmoid(self.fc2(out))

        return x * out[:, :, None]


class PoseHighResolutionNet(nn.Module):
    # 到时候这些参数应该都能放到cfg文件里面去
    def __init__(self, cfg, adj, p_dropout, gcn_type, nodes_group):
        super(PoseHighResolutionNet, self).__init__()
        self.inplanes = 32

        self.adj = [adj, adj_mx_from_edges(68, [[0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [3, 7], [3, 6], [6, 8], [7, 9],
                                                [8, 10], [9, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
                                                , [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25]
                                                , [2, 26], [2, 27], [2, 28], [2, 29], [2, 30], [2, 31], [2, 32], [2, 33]
                                                , [2, 34], [2, 35], [2, 36], [2, 37], [2, 38], [2, 39], [2, 40], [2, 41]
                                                , [2, 42], [2, 43], [2, 44], [2, 45], [5, 46], [46, 47], [47, 48]
                                                , [46, 49], [49, 50], [46, 51], [51, 52], [46, 53], [53, 54], [46, 55]
                                                , [55, 56], [4, 57], [57, 58], [58, 59], [57, 60], [60, 61], [57, 62]
                                                , [62, 63], [57, 64], [64, 65], [57, 66], [66, 67]], sparse=False),
                    adj_mx_from_edges(34, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [1, 7], [1, 8], [1, 9]
                                                 , [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
                                                 , [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [2, 23], [23, 24]
                                                 , [24, 25], [23, 26], [26, 27], [23, 28], [28, 29], [23, 30], [30, 31]
                                                 , [23, 32], [32, 33]], sparse=False)]

        self.p_dropout = p_dropout
        self.gcn_type = gcn_type
        self.nodes_group = nodes_group
        self.Sigmoid = nn.Sigmoid()
        extra = cfg['MODEL']['EXTRA']

        # stem net
        self.gconv1 = _GraphConv(adj, 2, 32, p_dropout, gcn_type)

        self.gconv2 = _GraphConv(adj, 32, 32, p_dropout, gcn_type)

        self.layer1 = self._make_layer(BasicBlock, 32, 1)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([32], num_channels)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, multi_scale_output=False)
        self.se_blocks = SEBlock(adj, pre_stage_channels[0])
        self.gconv_output = nn.Conv1d(pre_stage_channels[0], 3, (1,))

    # 这个函数工作仅仅是从前一个stage的最后一个branch下采样出后一个 stage的新的branch，图示中在transition处有多层融合不知道是如何实现的
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 改变通道即可
                    transition_layers.append(
                        nn.Sequential(
                            _GraphConv(self.adj[i],
                                       num_channels_pre_layer[i],
                                       num_channels_cur_layer[i],
                                       self.p_dropout, self.gcn_type
                                       ),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                downs = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    downs.append(
                        nn.Sequential(
                            _GraphConv(self.adj[i - 1],
                                       inchannels,
                                       outchannels,
                                       self.p_dropout,
                                       self.gcn_type
                                       ),
                            _SkeletalPool(self.nodes_group)  # 应该只需要传 最大的骨骼图就行了，再者这个类也是nn.moudle的子类，应该能放到sequential里面

                        )
                    )
                transition_layers.append(nn.Sequential(*downs))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        pooling = False
        channel_change = False
        if stride != 1:
            pooling = True
        if self.inplanes != planes * block.expansion:  # 64!=64*2
            channel_change = True

        layers = [block(self.adj[0], self.inplanes, planes, self.p_dropout, self.gcn_type, self.nodes_group, pooling,
                        channel_change)]
        self.inplanes = planes * block.expansion  # self.inplanes=64*2=128
        for i in range(1, blocks):
            layers.append(block(self.adj[0], self.inplanes, planes, self.p_dropout, self.gcn_type, self.nodes_group))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']  # 做特征融合的次数
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    self.adj,
                    self.nodes_group,
                    self.p_dropout,
                    self.gcn_type,
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.gconv1(x)

        x = self.gconv2(x)

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        out = self.Sigmoid(y_list[0].transpose(1, 2))
        out = self.gconv_output(out).transpose(1, 2)

        return out[:, :23, :], out[:, 23:91, :], out[:, 91:112, :], out[:, 112:, :]

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




def get_pose_net(cfg, is_train, adj, p_dropout, gcn_type, nodes_group):
    model = PoseHighResolutionNet(cfg, adj, p_dropout, gcn_type, nodes_group)
    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights()

    return model
