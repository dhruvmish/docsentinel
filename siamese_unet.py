import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # pad to same size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, out_ch=1):
        super().__init__()

        self.encoder = Encoder(in_ch, base_ch)

        # After concatenation channel sizes double
        self.up1 = Up(base_ch * 8 + base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2 + base_ch * 2, base_ch)
        self.up4 = Up(base_ch + base_ch, base_ch)

        self.outc = OutConv(base_ch, out_ch)

    def forward(self, imgA, imgB):
        x1a, x2a, x3a, x4a, x5a = self.encoder(imgA)
        x1b, x2b, x3b, x4b, x5b = self.encoder(imgB)

        d1 = torch.abs(x1a - x1b)
        d2 = torch.abs(x2a - x2b)
        d3 = torch.abs(x3a - x3b)
        d4 = torch.abs(x4a - x4b)
        d5 = torch.abs(x5a - x5b)

        x = self.up1(d5, d4)
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)

        return self.outc(x)


if __name__ == "__main__":
    model = SiameseUNet()
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    out = model(x, y)
    print("Output shape:", out.shape)
