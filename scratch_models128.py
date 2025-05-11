import torch.nn as nn
import torch.nn.functional as F

class SimpleNet1(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 128 128 3 -> 128 128 64 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 128 128 64 -> 64 64 64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 64 64 -> 64 64 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 64 64 128 -> 32 32 128
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 32 32 128 -> 32 32 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32 32 256 -> 16 16 256
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 16 16 256 -> 16 16 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16 16 512 -> 8 8 512
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 8 8 512 -> 1 1 512
            nn.Flatten(), # 1 1 512 -> 512
            nn.Linear(512, 256), # 512 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Dropout layer
            nn.Linear(256, 128), # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Dropout layer
            nn.Linear(128, num_classes), # 128 -> num_classes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class SimpleNet2(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), # 128x128x3 → 128x128x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), # 128x128x64 → 128x128x64
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2), # 128x128x64 → 64x64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), # 64x64x64 → 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 64x64x128 → 64x64x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2), # 64x64x128 → 32x32x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 32x32x128 → 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 32x32x256 → 32x32x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2), # 32x32x256 → 16x16x256

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), # 16x16x256 → 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), # 16x16x512 → 16x16x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16x512 → 8x8x512

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False), # 8x8x512 → 8x8x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2) # 8x8x256 → 4x4x256
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 4x4x256 = 4096
            nn.Linear(256 * 4 * 4, 512),   # 4096 → 512
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)   # 512 → num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class SimpleNet3(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2), # 128 128 3 -> 63 63 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 63 63 64 -> 31 31 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 31 31 64 -> 14 14 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #   14 14 128 -> 7 7 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 7 7 128 -> 3 3 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 288),
            nn.ReLU(inplace=True),
            nn.Linear(288, 144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 72),
            nn.ReLU(inplace=True),
            nn.Linear(72, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 18),
            nn.ReLU(inplace=True),
            nn.Linear(18, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
class SimpleNet4(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 128 128 3 -> 128 128 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 128 128 32 -> 64 64 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 64 64 32 -> 64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64 64 64 -> 32 32 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 32 64 -> 16 16 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16 16 128 -> 8 8 128
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),#   -> 4 x 4 x 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x