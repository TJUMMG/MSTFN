import torch
from torch import nn

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.rank_loss = RankLoss()

    def forward(self, out_images, target_images):
        # Content Loss
        image_loss = self.l1_loss(out_images, target_images)
        # Ranking Loss
        # TV Loss
        return image_loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.l2_loss = nn.MSELoss()
        # self.rank_loss = RankLoss()

    def forward(self, out_images, target_images):
        # Content Loss
        image_loss = self.l2_loss(out_images, target_images)
        # Ranking Loss
        # TV Loss
        return image_loss

class G_advsarialLoss(nn.Module):
    def __init__(self):
        super(G_advsarialLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.rank_loss = RankLoss()

    def forward(self, out_labels):
        # Adversarial Loss
        g_adversarial_loss = -out_labels
        # Ranking Loss
        # TV Loss
        return g_adversarial_loss

class D_advsarialLoss(nn.Module):
    def __init__(self):
        super(D_advsarialLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        # self.rank_loss = RankLoss()

    def forward(self, fake_out_label, real_out_label):
        # Adversarial Loss
        d_adversarial_loss = fake_out_label - real_out_label
        # Ranking Loss
        # TV Loss
        return d_adversarial_loss

class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.l1_loss = nn.MSELoss()
        self.c_matrix = c_matrix()

    def forward(self, output, target):

        output_c = self.c_matrix(output)
        target_c = self.c_matrix(target)
        c_loss = self.l1_loss(output_c, target_c)

        return c_loss

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin = 0.000001, p = 2)
        self.gram = gram_matrix()

    def forward(self, input, out, target):
        input_g = self.gram(input)

        out_g = self.gram(out)

        target_g = self.gram(target)

        # d_ot = self.L1_loss(out_g,target_g)
        # d_ox = self.L1_loss(out_g,input_g)
        triplet_loss = self.triplet_loss(out_g, target_g, input_g)
        # print("1",triplet_loss)
        # print("2",d_ot-d_ox)

        return triplet_loss

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gram = gram_matrix()

    def forward(self, input, out, target):
        input_g = self.gram(input)

        out_g = self.gram(out)

        target_g = self.gram(target)

        # f1 = self.dst(out_g, target_g)
        # f2 = self.dst(out_g, x_g)
        d_ot = self.L1_loss(out_g,target_g)
        d_ox = self.L1_loss(out_g,input_g)
        e_ot = torch.exp(-d_ot)
        e_ox = torch.exp(-d_ox)
        rank_loss = -torch.log(e_ot/(e_ot+e_ox))
        # rank_loss = d_ot - d_ox

        return rank_loss

class gram_matrix(nn.Module):
    def forward(self, input):
        batches,channels,height,width = input.size()
        feature = input.view(batches, channels, height*width)
        feature_T = feature.permute(0,2,1)
        gram= feature.bmm(feature_T)
        return gram.div(channels*height*width)

class c_matrix(nn.Module):
    def forward(self, input):
        batches,channels,height,width = input.size()
        feature = input.view(batches*channels, height*width)
        feature_T = feature.permute(1, 0)
        gram= torch.mm(feature_T, feature)
        return gram

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_h = self.tensor_size(x[:, :, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, :, 1:])
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4]


