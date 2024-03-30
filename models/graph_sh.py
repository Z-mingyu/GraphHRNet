from __future__ import absolute_import
import torch
import torch.nn as nn
from functools import reduce
from common.graph_utils import adj_mx_from_edges

from models.gconv.vanilla_graph_conv import DecoupleVanillaGraphConv
from models.gconv.pre_agg_graph_conv import DecouplePreAggGraphConv
from models.gconv.post_agg_graph_conv import DecouplePostAggGraphConv
from models.gconv.conv_style_graph_conv import ConvStyleGraphConv
from models.gconv.no_sharing_graph_conv import NoSharingGraphConv
from models.gconv.modulated_gcn_conv import ModulatedGraphConv
from models.gconv.sem_graph_conv import SemGraphConv

from models.graph_non_local import GraphNonLocal


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


class _Hourglass(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim1, hid_dim2, nodes_group, p_dropout, gcn_type):
        super(_Hourglass, self).__init__()

        # adj_mid = adj_mx_from_edges(68, [[0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [3, 7], [3, 6], [6, 8], [7, 9],
        #                                  [8, 10], [9, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
        #     , [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25]
        #     , [2, 26], [2, 27], [2, 28], [2, 29], [2, 30], [2, 31], [2, 32], [2, 33]
        #     , [2, 34], [2, 35], [2, 36], [2, 37], [2, 38], [2, 39], [2, 40], [2, 41]
        #     , [2, 42], [2, 43], [2, 44], [2, 45], [5, 46], [46, 47], [47, 48]
        #     , [46, 49], [49, 50], [46, 51], [51, 52], [46, 53], [53, 54], [46, 55]
        #     , [55, 56], [4, 57], [57, 58], [58, 59], [57, 60], [60, 61], [57, 62]
        #     , [62, 63], [57, 64], [64, 65], [57, 66], [66, 67]], sparse=False)
        # adj_low = adj_mx_from_edges(35, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [1, 7], [1, 8], [1, 9]
        #     , [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
        #     , [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [2, 23], [23, 24]
        #     , [23, 25], [23, 26], [23, 27], [23, 28], [2, 29], [29, 30], [29, 31], [29, 32]
        #     , [29, 33], [29, 34]], sparse=False)

        adj_mid = adj_mx_from_edges(68, [[0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [3, 7], [3, 6], [6, 8], [7, 9],
                                                [8, 10], [9, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17]
                                                , [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25]
                                                , [2, 26], [2, 27], [2, 28], [2, 29], [2, 30], [2, 31], [2, 32], [2, 33]
                                                , [2, 34], [2, 35], [2, 36], [2, 37], [2, 38], [2, 39], [2, 40], [2, 41]
                                                , [2, 42], [2, 43], [2, 44], [2, 45], [5, 46], [46, 47], [47, 48]
                                                , [46, 49], [49, 50], [46, 51], [51, 52], [46, 53], [53, 54], [46, 55]
                                                , [55, 56], [4, 57], [57, 58], [58, 59], [57, 60], [60, 61], [57, 62]
                                                , [62, 63], [57, 64], [64, 65], [57, 66], [66, 67]], sparse=False)
        adj_low = adj_mx_from_edges(34, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [1, 7], [1, 8], [1, 9]
                                                 , [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17]
                                                 , [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [2, 23], [23, 24]
                                                 , [24, 25], [23, 26], [26, 27], [23, 28], [28, 29], [23, 30], [30, 31]
                                                 , [23, 32], [32, 33]], sparse=False)

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim1, p_dropout, gcn_type)
        self.gconv2 = _GraphConv(adj_mid, hid_dim1, hid_dim2, p_dropout, gcn_type)
        self.gconv3 = _GraphConv(adj_low, hid_dim2, hid_dim2, p_dropout, gcn_type)
        self.gconv4 = _GraphConv(adj_mid, hid_dim2, hid_dim1, p_dropout, gcn_type)
        self.gconv5 = _GraphConv(adj, hid_dim1, output_dim, p_dropout, gcn_type)

        self.pool = _SkeletalPool(nodes_group)
        self.unpool = _SkeletalUnpool()

    def forward(self, x):
        skip1 = x
        skip2 = self.gconv1(skip1)
        skip3 = self.gconv2(self.pool(skip2))
        out = self.gconv3(self.pool(skip3))
        out = self.gconv4(self.unpool(out) + skip3)
        out = self.gconv5(self.unpool(out) + skip2)
        return out + skip1


# class _SkeletalPool(nn.Module):
#     def __init__(self, nodes_group):  # nodes_group是节点编号组合
#         super(_SkeletalPool, self).__init__()
#         self.high_group = sum(nodes_group, [])  # 多个列表合成一个
#         self.mid_group = [0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10, 11, 12, 20, 13, 19, 14, 18, 15, 17, 16, 38, 21,
#                           25, 22, 24, 23, 26, 27, 29, 28, 30, 31, 34, 32, 35, 33, 36, 37, 39, 40, 41, 42, 43, 44, 45,
#                           46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
#                           67]
#         # 如果是whole_body，body+feet（23->12->6）,hand(21->11->6),face(68->34->17)
#         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         if x.shape[1] == 133:  # x.shape=[batch,node_nums,token_size]
#             out = self.pool(x[:, self.high_group].transpose(1, 2))
#             return out.transpose(1, 2)
#         elif x.shape[1] == 68:
#             out = self.pool(x[:, self.mid_group].transpose(1, 2))  # 第二级是严格按照顺序的，那么可以直接相邻的节点特征做pooling
#             return out.transpose(1, 2)
#         else:
#             assert False, 'Invalid Type in Skeletal Pooling : x.shape is {}'.format(x.shape)
#
#
# class _SkeletalUnpool(nn.Module):
#     def __init__(self):
#         super(_SkeletalUnpool, self).__init__()
#         self.inv_low = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 11, 12, 13, 12, 11, 13, 14, 15,
#                         14, 15, 16, 17, 18, 16, 17, 18, 19, 10, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25, 25, 26, 26,
#                         27, 27, 28, 28, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34]
#
#         self.inv_mid = [2, 0, 0, 1, 1, 3, 3, 5, 4, 5, 4, 7, 6, 7, 6, 9, 8, 11, 11, 9, 10, 10, 8, 12, 12, 13, 13, 14, 14,
#                         15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27,
#                         27, 28, 28, 29, 30, 30, 31, 32, 32, 31, 33, 33, 34, 35, 35, 34, 36, 36, 37, 37, 38, 29, 38, 39,
#                         39, 40, 40, 16, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51,
#                         51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63,
#                         64, 64, 65, 65, 66, 66, 67, 67]  # self.inv_mid[80]=16,self.inv_mid[74]=29直接给出索引对应关系得到上采样的结果
#
#     def forward(self, x):
#         if x.shape[1] == 68:
#             return x[:, self.inv_mid]
#         elif x.shape[1] == 35:
#             return x[:, self.inv_low]
#         else:
#             assert False, 'Invalid Type in Skeletal Unpooling : x.shape is {}'.format(x.shape)


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

class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


# 做全局特征提取的
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


class GraphSH(nn.Module):
    def __init__(self, adj, hid_dim, nodes_group, coords_dim=(2, 3), num_layers=4, p_dropout=None, gcn_type=None):
        super(GraphSH, self).__init__()

        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout, gcn_type=gcn_type)
        self.num_layers = num_layers
        _gconv_layers = []
        _conv_layers = []

        group_size = len(nodes_group[0])  # 8
        assert group_size > 1

        grouped_order = list(reduce(lambda x, y: x + y,
                                    nodes_group))  # 把[[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]合成一个列表
        restored_order = [0] * len(grouped_order)
        # 把每个节点在grouped_order的索引存在restored_order中，但是很奇怪的是后面没有用到这个东西
        for i in range(len(restored_order)):
            for j in range(len(grouped_order)):
                if grouped_order[j] == i:
                    restored_order[i] = j
                    break

        for i in range(num_layers):
            _gconv_layers.append(
                _Hourglass(adj, hid_dim, hid_dim, int(hid_dim * 1.5), hid_dim * 2, nodes_group, p_dropout, gcn_type))
            _conv_layers.append(nn.Conv1d(hid_dim, hid_dim // num_layers, 1))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.conv_layers = nn.ModuleList(_conv_layers)

        self.se_blocks = SEBlock(adj, hid_dim)
        self.gconv_output = nn.Conv1d(hid_dim, coords_dim[1], 1)

    def forward(self, x):
        out = self.gconv_input(x)
        inter_fs = []
        for l in range(self.num_layers):
            out = self.gconv_layers[l](out)
            inter_fs.append(self.conv_layers[l](out.transpose(1, 2)).transpose(1, 2))
        f_out = torch.cat(inter_fs, dim=2)
        #out = self.se_blocks(f_out.transpose(1, 2))
        out = self.gconv_output(f_out.transpose(1,2)).transpose(1, 2)
        return out[:, :23, :], out[:, 23:91, :], out[:, 91:112, :], out[:, 112:, :]
