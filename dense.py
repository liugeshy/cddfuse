import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        # self.norm = nn.LayerNorm([in_channels, 128, 128])  # Assuming input size is [224, 224]
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

        self.gelu = nn.GELU()

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.gelu(x1+x)

        x2 = self.conv2(x1)
        x2 = self.gelu(x2+x1+x)

        x3 = self.conv3(x2)
        x3 = self.gelu(x3+x2+x1+x)

        x4 = self.conv4(x3)
        x4 = self.gelu(x4+x3+x2+x1+x)

        x5 = self.conv5(x4)
        x5 = self.gelu(x5+x4+x3+x2+x1+x)

        x6= self.conv6(x5)
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x)

        return x6
    

if __name__ == '__main__':
    # 创建一个随机输入张量
    x = torch.randn(1, 32, 16, 16)  # batch_size=1, channels=32, height=16, width=16

    # 创建模型实例
    model = Dense(in_channels=32)

    # 前向传播
    output = model(x)

    # 打印输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)