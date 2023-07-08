import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


# 时空特征融合残差块
# ==================================================================== #

class ProposedBlock_start(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(ProposedBlock_start, self).__init__()

        self.conv1_0 = Conv(in_ch, 64)
        self.conv1_1 = Conv(in_ch, 64)
        self.conv1_2 = Conv(in_ch, 64)

        self.conv2_0 = Conv(64, 64)
        self.conv2_1 = Conv(64, 64)
        self.conv2_2 = Conv(64, 64)

        self.one0 = Conv(64, 1)
        self.one1 = Conv(64, 1)
        self.one2 = Conv(64, 1)

        self.middle = Conv(192, 64)

        self.conv3_0 = Conv(64, 64)
        self.conv3_1 = Conv(64, 64)
        self.conv3_2 = Conv(64, 64)

        # self.conv4_0 = Conv(64, inter_ch)
        # self.conv4_1 = Conv(64, inter_ch)
        # self.conv4_2 = Conv(64, inter_ch)
        # self.conv4_3 = Conv(64, inter_ch)
        # self.conv4_4 = Conv(64, inter_ch)

    def forward(self, x0, x1, x2):
        # 1st conv
        c1_fr0 = self.conv1_0(x0)
        c1_fr1 = self.conv1_1(x1)
        c1_fr2 = self.conv1_2(x2)

        # 2nd conv
        c2_fr0 = self.conv2_0(c1_fr0)
        c2_f0 = self.one0(c2_fr0)

        c2_fr1 = self.conv2_1(c1_fr1)
        c2_f1 = self.one1(c2_fr1)

        c2_fr2 = self.conv2_2(c1_fr2)
        c2_f2 = self.one2(c2_fr2)

        # 64fetures list # b,64,h,w
        c2_list = [c2_fr0, c2_fr1, c2_fr2]

        c2_fcat = [c2_f0, c2_f1, c2_f2]
        #  single channel feature concat
        c2_fcat = torch.cat(c2_fcat, dim=1)  # b,3,h,w
        #  time attention map
        time_att = F.softmax(c2_fcat, dim=1)

        # middle
        c2_att = [time_att[:, i, :, :] * c2_list[i] for i in range(3)]

        c2_concat = torch.cat(c2_list, dim=1)  # b, 192, h, w
        c2_fuse = self.middle(c2_concat)  # b, 64, h, w

        # 3rd conv
        c3_fr0 = self.conv3_0(c2_att[0] + c2_fuse)
        c3_fr1 = self.conv3_1(c2_att[1] + c2_fuse)
        c3_fr2 = self.conv3_2(c2_att[2] + c2_fuse)

        # 4th conv
        # c4_fr0 = self.conv4_0(c3_fr0)
        # c4_fr1 = self.conv4_1(c3_fr1)
        # c4_fr2 = self.conv4_2(c3_fr2)
        # c4_fr3 = self.conv4_3(c3_fr3)
        # c4_fr4 = self.conv4_4(c3_fr4)

        return c3_fr0, c3_fr1, c3_fr2


class ProposedBlock(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(ProposedBlock, self).__init__()

        self.conv1_0 = Conv(in_ch, 64)
        self.conv1_1 = Conv(in_ch, 64)
        self.conv1_2 = Conv(in_ch, 64)

        self.conv2_0 = Conv(64, 64)
        self.conv2_1 = Conv(64, 64)
        self.conv2_2 = Conv(64, 64)

        self.one0 = Conv(64, 1)
        self.one1 = Conv(64, 1)
        self.one2 = Conv(64, 1)

        self.middle = Conv(192, 64)

        self.conv3_0 = Conv(64, 64)
        self.conv3_1 = Conv(64, 64)
        self.conv3_2 = Conv(64, 64)

        # self.conv4_0 = Conv(64, inter_ch)
        # self.conv4_1 = Conv(64, inter_ch)
        # self.conv4_2 = Conv(64, inter_ch)
        # self.conv4_3 = Conv(64, inter_ch)
        # self.conv4_4 = Conv(64, inter_ch)

    def forward(self, x0, x1, x2):
        feature_map = []

        # 1st conv
        c1_fr0 = self.conv1_0(x0)
        c1_fr1 = self.conv1_1(x1)
        c1_fr2 = self.conv1_2(x2)

        # 2nd conv
        c2_fr0 = self.conv2_0(c1_fr0)
        c2_f0 = self.one0(c2_fr0)

        c2_fr1 = self.conv2_1(c1_fr1)
        c2_f1 = self.one1(c2_fr1)

        c2_fr2 = self.conv2_2(c1_fr2)
        c2_f2 = self.one2(c2_fr2)

        # 64fetures list # b,64,h,w
        c2_list = [c2_fr0, c2_fr1, c2_fr2]

        c2_fcat = [c2_f0, c2_f1, c2_f2]
        #  single channel feature concat
        c2_fcat = torch.cat(c2_fcat, dim=1)  # b,3,h,w
        #  time attention map
        time_att = F.softmax(c2_fcat, dim=1)

        # middle
        c2_att = [time_att[:, i, :, :]*c2_list[i] for i in range(3)]

        c2_concat = torch.cat(c2_list, dim=1)  # b, 192, h, w
        c2_fuse = self.middle(c2_concat)  # b, 64, h, w

        # 3rd conv
        c3_fr0 = self.conv3_0(c2_att[0]+c2_fuse)
        c3_fr1 = self.conv3_1(c2_att[1]+c2_fuse)
        c3_fr2 = self.conv3_2(c2_att[2]+c2_fuse)

        # 4th conv
        # c4_fr0 = self.conv4_0(c3_fr0)
        # c4_fr1 = self.conv4_1(c3_fr1)
        # c4_fr2 = self.conv4_2(c3_fr2)
        # c4_fr3 = self.conv4_3(c3_fr3)
        # c4_fr4 = self.conv4_4(c3_fr4)

        return c3_fr0+x0, c3_fr1+x1, c3_fr2+x2


# 返回中间的特征图，和原函数没有什么差别，除了返回值
class ProposedBlock_feature(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super(ProposedBlock_feature, self).__init__()

        self.conv1_0 = Conv(in_ch, 64)
        self.conv1_1 = Conv(in_ch, 64)
        self.conv1_2 = Conv(in_ch, 64)

        self.conv2_0 = Conv(64, 64)
        self.conv2_1 = Conv(64, 64)
        self.conv2_2 = Conv(64, 64)

        self.one0 = Conv(64, 1)
        self.one1 = Conv(64, 1)
        self.one2 = Conv(64, 1)

        self.middle = Conv(192, 64)

        self.conv3_0 = Conv(64, 64)
        self.conv3_1 = Conv(64, 64)
        self.conv3_2 = Conv(64, 64)

        # self.conv4_0 = Conv(64, inter_ch)
        # self.conv4_1 = Conv(64, inter_ch)
        # self.conv4_2 = Conv(64, inter_ch)
        # self.conv4_3 = Conv(64, inter_ch)
        # self.conv4_4 = Conv(64, inter_ch)

    def forward(self, x0, x1, x2):
        feature_map = []

        # 1st conv
        c1_fr0 = self.conv1_0(x0)
        c1_fr1 = self.conv1_1(x1)
        c1_fr2 = self.conv1_2(x2)

        # 2nd conv
        c2_fr0 = self.conv2_0(c1_fr0)
        c2_f0 = self.one0(c2_fr0)

        c2_fr1 = self.conv2_1(c1_fr1)
        c2_f1 = self.one1(c2_fr1)

        c2_fr2 = self.conv2_2(c1_fr2)
        c2_f2 = self.one2(c2_fr2)

        # 64fetures list # b,64,h,w
        c2_list = [c2_fr0, c2_fr1, c2_fr2]

        c2_fcat = [c2_f0, c2_f1, c2_f2]
        #  single channel feature concat
        c2_fcat = torch.cat(c2_fcat, dim=1)  # b,3,h,w
        #  time attention map
        time_att = F.softmax(c2_fcat, dim=1)

        feature_map = [torch.squeeze(torch.mean(c2_fr1, dim=1, keepdim=True), dim=0), torch.squeeze(
            torch.mean(time_att[:, 1, :, :]*c2_fr1, dim=1, keepdim=True), dim=0)]

        # middle
        c2_att = [time_att[:, i, :, :]*c2_list[i] for i in range(3)]

        c2_concat = torch.cat(c2_list, dim=1)  # b, 192, h, w
        c2_fuse = self.middle(c2_concat)  # b, 64, h, w

        feature_map.append(torch.squeeze(torch.mean(c2_fuse, dim=1, keepdim=True), dim=0))

        # 3rd conv
        c3_fr0 = self.conv3_0(c2_att[0]+c2_fuse)
        c3_fr1 = self.conv3_1(c2_att[1]+c2_fuse)
        c3_fr2 = self.conv3_2(c2_att[2]+c2_fuse)

        # 4th conv
        # c4_fr0 = self.conv4_0(c3_fr0)
        # c4_fr1 = self.conv4_1(c3_fr1)
        # c4_fr2 = self.conv4_2(c3_fr2)
        # c4_fr3 = self.conv4_3(c3_fr3)
        # c4_fr4 = self.conv4_4(c3_fr4)

        return c3_fr0+x0, c3_fr1+x1, c3_fr2+x2, feature_map
