import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

    def forward(self, input):
        return self.conv(input)

class DeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, input):
        return self.deconv(input)

class Conv_wo_ReLu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_wo_ReLu, self).__init__()
        self.conv_nl = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, input):
        return self.conv_nl(input)