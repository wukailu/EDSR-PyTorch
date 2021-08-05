from torch import nn


class Plain_SR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, config=""):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if 'prelu' in config:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU()
        if "bn" in config:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.skip = (in_channels == out_channels)

    def forward(self, x):
        if self.skip:
            return self.act(self.norm(self.conv(x)) + x)
        else:
            return self.act(self.norm(self.conv(x)))


class Plain_SR2_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, config=""):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if 'prelu' in config:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU()
        if "bn" in config:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.skip = (in_channels == out_channels)

    def forward(self, x):
        if self.skip:
            return self.act(self.norm(self.conv(x))) + x
        else:
            return self.act(self.norm(self.conv(x)))
