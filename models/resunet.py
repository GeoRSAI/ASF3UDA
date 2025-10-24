import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(ResNetEncoder, self).__init__()
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)

        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))

        # Store the layers to extract intermediate features
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.layer1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x):
        # Extract intermediate features for skip connections
        x0 = self.layer0(x)  # Initial conv layer
        x1 = self.layer1(x0) # Layer 1
        x2 = self.layer2(x1) # Layer 2
        x3 = self.layer3(x2) # Layer 3
        x4 = self.layer4(x3) # Layer 4
        return x0, x1, x2, x3, x4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(ResUNet, self).__init__()
        self.encoder = ResNetEncoder(backbone, pretrained)

        # Decoder blocks with skip connections
        self.decoder1 = DecoderBlock(2048 + 1024, 1024)  # Layer4 + Layer3
        self.decoder2 = DecoderBlock(1024 + 512, 512)    # Layer3 + Layer2
        self.decoder3 = DecoderBlock(512 + 256, 256)      # Layer2 + Layer1
        self.decoder4 = DecoderBlock(256 + 64, 64)        # Layer1 + Layer0

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution to get the desired number of classes
        self.final_conv = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Decoder with skip connections
        d1 = self.decoder1(x4, x3)  # Layer4 + Layer3
        d2 = self.decoder2(d1, x2)  # Layer3 + Layer2
        d3 = self.decoder3(d2, x1)  # Layer2 + Layer1
        d4 = self.decoder4(d3, x0)  # Layer1 + Layer0
        d4 = self.upsample(d4)
        # Final convolution
        out = self.final_conv(d4)
        return out

# Example usage
if __name__ == "__main__":
    model = ResUNet(backbone='resnet50', pretrained=True)
    input_tensor = torch.randn(2, 3, 256, 256)  # Example input tensor
    output = model(input_tensor)
    print(output.shape)  