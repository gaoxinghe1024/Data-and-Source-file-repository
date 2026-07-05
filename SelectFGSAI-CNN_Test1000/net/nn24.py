import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)

    def forward(self, x):
        return x * self.ca(x)

class DensityNet(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            Attention(64),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class BinaryNet(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class HistNet(nn.Module):
    def __init__(self, out_features=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=torch.cat([x1,x2], dim=1)
        x4=self.mlp(x3)
        return x4

class SelectFGSAI_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.density_net = DensityNet(out_features=128)
        self.binary_net = BinaryNet(out_features=128)
        self.hist_net = HistNet(out_features=64)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        density = x[:, 0:1, :, :]
        binary = x[:, 1:2, :, :]
        hist = x[:, 2:3, :, :]

        feat_density = self.density_net(density)
        feat_binary = self.binary_net(binary)
        feat_hist = self.hist_net(hist)

        combined = torch.cat([feat_density, feat_binary, feat_hist], dim=1)
        out = self.classifier(combined)
        return out

if __name__ == "__main__":
    input_vector = torch.randn(1, 3, 484, 484)
    model = SelectFGSAI_CNN(num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(input_vector)
    print("Output shape:", output.shape)
