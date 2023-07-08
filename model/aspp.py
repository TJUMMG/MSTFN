import torch
import torch.nn as nn


# 空洞空间卷积池化金字塔模块
# ======================================================================= #
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
			stride=1, padding=padding, dilation=rate, bias=False)
        self.prelu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)

        return self.prelu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=1)
        self.aspp2 = ASPP_module(inplanes, planes, rate=2)
        self.aspp3 = ASPP_module(inplanes, planes, rate=4)
        self.aspp4 = ASPP_module(inplanes, planes, rate=8)

        self.prelu = nn.ReLU()

        self.conv1 = nn.Conv2d(planes*4, planes, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.prelu(x)
        return x

# ======================================================================= #