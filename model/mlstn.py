import torch
import torch.nn as nn
from .layers import *
from .aspp import ASPP
from .rstf import ProposedBlock



class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.conv(x)
        res += x
        return res

class ResPBRes(nn.Module):
    def __init__(self):
        super(ResPBRes, self).__init__()
        self.res_head1 = ResBlock(64,64)
        self.res_head2 = ResBlock(64, 64)
        self.res_head3 = ResBlock(64, 64)
        self.pb = ProposedBlock(64,64)
        self.res_tail1 = ResBlock(64,64)
        self.res_tail2 = ResBlock(64, 64)
        self.res_tail3 = ResBlock(64, 64)

    def forward(self, x1,x2,x3):
        xx1 = self.res_head1(x1)
        xx2 = self.res_head2(x2)
        xx3 = self.res_head3(x3)
        xx_at1, xx_at2, xx_at3 = self.pb(xx1,xx2,xx3)
        xt_1 = self.res_tail1(xx_at1)
        xt_2 = self.res_tail2(xx_at2)
        xt_3 = self.res_tail3(xx_at3)

        return xt_1,xt_2,xt_3

class generalModule(nn.Module):
    def __init__(self,in_ch):
        super(generalModule, self).__init__()
        self.conv1 = Conv(in_ch,64)
        self.conv2 = Conv(in_ch,64)
        self.conv3 = Conv(in_ch,64)
        self.rpb1 = ResPBRes()
        self.rpb2 = ResPBRes()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.aspp = ASPP(64,64)

    def forward(self, x1, x2, x3):
        xc1 = self.conv1(x1)
        xc2 = self.conv2(x2)
        xc3 = self.conv3(x3)
        xx1,xx2,xx3 = self.rpb1(xc1,xc2,xc3)
        xxx1,xxx2,xxx3 = self.rpb2(xx1,xx2,xx3)
        xxx = xxx1 + xxx2 + xxx3
        x = self.conv(xxx)
        x_aspp = self.aspp(x)

        return x_aspp


class vbde_net(nn.Module):
    # def __init__(self,in_ch,out_ch):# 15,3
    def __init__(self):  # 15,3
        super(vbde_net, self).__init__()
        # conv ith_frame

        self.stage1 = generalModule(3)
        self.stage2 = generalModule(64)

        self.convtail1 = ResBlock(64, 64)
        self.convtail2 = ResBlock(64, 64)
        self.convtail3 = ResBlock(64, 64)
        self.convtail4 = Conv(64, 3)

    def forward(self, x):
        b, ct, h, w = x.size()

        # print(x.size())
        x_f = []  # storage i frame
        x_3 = x[:, 6:9, :, :]

        for i in range(5):  # i frame
            x_f.append(x[:, i * 3:(i + 1) * 3, :, :])
        # x_grp1 = x[:, 0:9, :, :]
        # x_grp2 = x[:, 3:12, :, :]
        # x_grp3 = x[:, 6:15, :, :]
        # stage1
        s1g1 = self.stage1(x_f[0], x_f[1], x_f[2])
        s1g2 = self.stage1(x_f[1], x_f[2], x_f[3])
        s1g3 = self.stage1(x_f[2], x_f[3], x_f[4])

        

        # stage2
        s2 = self.stage2(s1g1, s1g2, s1g3)

        


        # 3rd group


        # 2nd conv

        # 3rd conv
        # concat
        # c3_cat = [c3_fr0, c3_fr1, c3_fr2]
        # c3_cat = torch.cat(c3_cat, dim=1)
        # add

        # 4th conv
        # c4 = self.convtail1(c3)

        # 5th conv
        c4 = self.convtail1(s2)
        c5 = self.convtail2(c4)
        c6 = self.convtail3(c5)

        c7 = self.convtail4(c6)

        return c7+x_3