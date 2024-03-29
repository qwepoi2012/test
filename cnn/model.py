import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()

        #conv layers
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # linear layers
        input_size = 32*64*64
        self.linear1= nn.Linear(input_size, 256)
        self.linear2= nn.Linear(256, 64)
        self.linear3= nn.Linear(64, num_classes)

        #activation layers

        self.relu =nn.ReLU()

        self.conv_layers = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu)
        self.linear_layers = nn.Sequential(
            self.linear1,
            self.relu,
            nn.Dropout(0.5),
            self.linear2,
            self.relu,
            nn.Dropout(0.3),
            self.linear3)
        
    def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)

            return x

