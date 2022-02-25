import torch
from torch import nn
from models.resnet50 import resnet50
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.arm8 = AttentionRefinementModule(512, 256)
        self.arm16 = AttentionRefinementModule(1024, 256)
        self.conv_head8 = ConvBNReLU(256, 256, 3, 1, 1)
        self.conv_head16 = ConvBNReLU(256, 256, 3, 1, 1)
        self.conv_avg = ConvBNReLU(1024, 256, 1, 1, 0)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, feat4, feat8, feat16):
        H4, W4 = feat4.shape[2:]
        H8, W8 = feat8.shape[2:]
        H16, W16 = feat16.shape[2:]

        avg = F.avg_pool2d(feat16, feat16.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H16, W16), mode='nearest')

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + avg_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        feat8_arm = self.arm8(feat8)
        feat8_sum = feat8_arm + feat16_up
        feat8_up = F.interpolate(feat8_sum, (H4, W4), mode='nearest')
        feat8_up = self.conv_head8(feat8_up)

        return feat4, feat8_up, feat16_up


class BiSeNetOutPut(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class GatedConv2d(nn.Module):
    def __init__(self, in_chan, mid_chan):
        super().__init__()
        self.ConvMask = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, 3, 1, 1),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, in_chan, 3, 1, 1),
            nn.Sigmoid()
        )
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, feat):
        mask = self.ConvMask(feat)
        out = feat * mask
        return out


class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.feat = resnet50(pretrained=True)
        self.coeff = nn.Conv2d(2048, 257, (1, 1))
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(512, 256)
        self.conv_out = BiSeNetOutPut(256, 256, n_classes=1)
        self.conv_out8 = BiSeNetOutPut(256, 128, n_classes=1)
        self.conv_out16 = BiSeNetOutPut(256, 128, n_classes=1)
        self.gated = GatedConv2d(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.weight_init()

    @torch.no_grad()
    def weight_init(self):
        self.coeff.weight.zero_()
        self.coeff.bias.zero_()

    def forward(self, img):
        H, W = img.shape[2:]
        feat4, feat8, feat16 = self.feat(img)  # 1/4, 1/8. 1/16 of initial size
        feat_res4, feat_cp4, feat_cp8 = self.cp(feat4, feat8, feat16)
        feat_sp = feat_res4

        feat_fuse = self.ffm(feat_sp, feat_cp4)
        feat_out = self.conv_out(feat_fuse)
        feat_out8 = self.conv_out8(feat_cp4)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out8 = F.interpolate(feat_out8, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat16 = self.gated(feat16)
        feat = self.feat.layer4(feat16)
        feat = self.avgpool(feat)
        coeff = self.coeff(feat)
        return coeff.squeeze(2).squeeze(2), feat_out, feat_out8, feat_out16


if __name__ == '__main__':
    model = FaceEncoder()
    x = torch.rand(10, 3, 256, 256)
    out = model(x)
    for o in out:
        print(o.shape)
