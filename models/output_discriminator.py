import torch.nn as nn
import torch
from models.ftunetformer import *

class OutputDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(OutputDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=8, mode='bilinear')
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        x = self.sigmoid(x) 

        return x

class DConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(DConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding),
            norm_layer(out_channels)
        )

class DConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(DConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class DWF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(DWF, self).__init__()
        self.downsample = nn.Conv2d(in_channels, decode_channels, kernel_size=4, stride=2, padding=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x):
        x = self.downsample(x)
        x = self.post_conv(x)
        return x

class UWF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(UWF, self).__init__()
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        return x
    
class DBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class DFeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = DConvBNReLU(in_channels, decode_channels, kernel_size=4, stride=2, padding=1)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class transDiscri(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=1):
        super(transDiscri, self).__init__()

        # self.pre_conv = DConvBN(num_classes, decode_channels, kernel_size=4, stride=2, padding=1)
        self.pre_conv = DConvBN(num_classes, decode_channels, kernel_size=6, stride=4, padding=1)
        self.b4 = DBlock(dim=decode_channels, num_heads=16, window_size=window_size)

        self.p3 = DWF(decode_channels, decode_channels * 2)
        self.b3 = DBlock(dim=decode_channels * 2, num_heads=16, window_size=window_size)

        self.p2 = DWF(decode_channels * 2, decode_channels * 4)
        self.b2 = DBlock(dim=decode_channels * 4, num_heads=16, window_size=window_size)

        self.p1 = DFeatureRefinementHead(decode_channels * 4, decode_channels)
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, 1, kernel_size=1))
        self.init_weight()

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.b4(x)

        x = self.p3(x)
        x = self.b3(x)

        x = self.p2(x)
        x = self.b2(x)

        x = self.p1(x)
        x = self.segmentation_head(x) # 1*16*16

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    x = torch.rand(2, 1, 256, 256).cuda()
    m = transDiscri(num_classes=1).cuda()
    m.eval()
    y = m(x)
    print(y.shape)