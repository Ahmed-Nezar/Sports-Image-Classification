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
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),  # 224x224x3 -> 224x224x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), # 224x224x64 -> 224x224x64
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 224x224x64 -> 112x112x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),  # 112x112x64 -> 112x112x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 112x112x128 -> 112x112x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112x128 -> 56x56x128

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 56x56x128 -> 56x56x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 56x56x256 -> 56x56x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56x256 -> 28x28x256

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), # 28x28x256 -> 28x28x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), # 28x28x512 -> 28x28x512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28x512 -> 14x14x512

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False), # 14x14x512 -> 14x14x256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14x256 -> 7x7x256
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 7x7x256 -> 1x1x256
            nn.Flatten(),  # 1x1x256 -> 256
            nn.Linear(256, 512),  # 256 -> 512
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),  # 512 -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleNet3(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 224x224x3 -> 112x112x16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112x112x16 -> 56x56x16

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56x56x16 -> 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 28x28x32 -> 14x14x32

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14x32 -> 7x7x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 7x7x64 -> 1x1x64
            nn.Flatten(),  # 1x1x64 -> 64
            nn.Linear(64, 32),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),  # 16 -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleNet4(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleNet4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224x3 -> 224x224x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 224x224x32 -> 112x112x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 112x112x32 -> 112x112x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112x112x64 -> 56x56x64

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56x56x64 -> 28x28x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 28x28x128 -> 14x14x128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 14x14x128 -> 7x7x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 7x7x256 -> 1x1x256
            nn.Flatten(),  # 1x1x256 -> 256
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),  # 16 -> num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x