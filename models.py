## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, output_features):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # self.cnn_layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=5),
        #     # torch.nn.BatchNorm2d(32),
        #     torch.nn.Sigmoid(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     # torch.nn.Dropout(p=0.5),
        #     torch.nn.Conv2d(32, 64, kernel_size=5),
        #     # torch.nn.BatchNorm2d(64),
        #     torch.nn.Sigmoid(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     # torch.nn.Dropout(p=0.5),
        #     torch.nn.Conv2d(64, 128, kernel_size=3),
        #     # torch.nn.BatchNorm2d(128),
        #     torch.nn.Sigmoid(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     # torch.nn.Dropout(p=0.5),
        #     torch.nn.Conv2d(128, 256, kernel_size=3),
        #     # torch.nn.BatchNorm2d(256),
        #     torch.nn.Sigmoid(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     # torch.nn.Dropout(p=0.5),
        #     torch.nn.Conv2d(256, 512, kernel_size=3),
        #     # torch.nn.BatchNorm1d(512),
        #     torch.nn.Sigmoid(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     # torch.nn.Dropout(p=0.5),
        # )
        # self.cnn_layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=5),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(32, 64, kernel_size=3),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(64, 128, kernel_size=3),
        #     torch.nn.Conv2d(128, 128, kernel_size=3),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(128, 256, kernel_size=3),
        #     torch.nn.Conv2d(256, 256, kernel_size=3),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        #     torch.nn.Conv2d(256, 512, kernel_size=3),
        #     torch.nn.Conv2d(512, 512, kernel_size=3)
        # )

        # self.fc_layers = torch.nn.Sequential(
        #     torch.nn.Linear(18432, 1000),
        #     torch.nn.Dropout(p=0.4),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Linear(1000, 500),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Dropout(p=0.4),
        #     torch.nn.Linear(500, 136),
        # )
        # self.cnn_layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=3),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.AvgPool2d(kernel_size=2),

        #     torch.nn.Conv2d(32, 64, kernel_size=3),
        #     torch.nn.BatchNorm2d(64),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.AvgPool2d(kernel_size=2),

        #     torch.nn.Conv2d(64, 128, kernel_size=3),
        #     torch.nn.BatchNorm2d(128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.AvgPool2d(kernel_size=2),

        #     torch.nn.Conv2d(128, 256, kernel_size=3),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.AvgPool2d(kernel_size=2),

        #     torch.nn.Conv2d(256, 512, kernel_size=3),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.AvgPool2d(kernel_size=2),
        # )
        # self.fc_layers = torch.nn.Sequential(
        #     torch.nn.Linear(4608, 2000),
        #     torch.nn.ReLU(0.4),
        #     torch.nn.Linear(2000, 1000),
        #     torch.nn.Dropout(0.4),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Linear(1000, 500),
        #     torch.nn.Linear(500, 136)
        # )
        # working model : 224x224
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(128, 256, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(256, 512, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(12800, 1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(500, 136),
        )

        # # working model 96x96
        # self.cnn_layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=5),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),

        #     torch.nn.Conv2d(32, 64, kernel_size=5),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),

        #     torch.nn.Conv2d(64, 128, kernel_size=3),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),

        #     torch.nn.Conv2d(128, 256, kernel_size=3),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),

        #     torch.nn.Conv2d(256, 512, kernel_size=1),
        #     torch.nn.Tanh(),
        #     torch.nn.MaxPool2d(kernel_size=2),
        # )
        # self.fc_layers = torch.nn.Sequential(
        #     torch.nn.Linear(512, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.4),
        #     torch.nn.Linear(512, 250),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.4),
        #     torch.nn.Linear(250, 136),
        # )

        
    def forward(self, in_tensor):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        in_tensor = self.cnn_layers(in_tensor)
        feature_tensor = torch.flatten(in_tensor, start_dim=1)
        logits = self.fc_layers(feature_tensor) # raw probabilites
        return logits
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        # return x
