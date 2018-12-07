import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniNet(nn.Module):
    # expected input (n, 1, 4, 4)
    def __init__(self):
        super(MiniNet, self).__init__()

        self.conv = nn.Conv2d(1, 3, kernel_size=(3,3), bias=False)
        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dense = nn.Linear(3, 2, bias=False)
        self.a = nn.ReLU()

    def forward(self, x):
        n = x.size(0)
        x = self.a(self.conv(x))
        x = self.mp(x)
        x = x.view(n, -1)
        x = self.dense(x)
        return x

class CNN_iNNvestigate(nn.Module):
    # a pytorch network that mimics the keras network the authors of iNNvestigate
    # define in the example notebook on mnist they have released on their github

    # it has no shallow max pooling and a large dense layer

    def __init__(self):
        super(CNN_iNNvestigate, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.dense1 = nn.Linear(9216, 512)
        self.dense2 = nn.Linear(512, 10)

        self.a = nn.ReLU()
        self.mp = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.a(self.conv1(x))
        x = self.a(self.conv2(x))
        x = self.mp(x)
        x = x.view(batch_size, -1)
        x = self.a(self.dense1(x))
        x = self.dense2(x)
        return x

class CNN(nn.Module):
    def __init__(self, init_depth=16):
        super(CNN, self).__init__()

        d = {}
        for i in range(5):
            d[i] = init_depth * 2**i
        self.conv1 = nn.Conv2d( 1, d[0], kernel_size=(3,3) )
        self.conv2 = nn.Conv2d( d[0], d[1], kernel_size=(3,3) )
        self.conv3 = nn.Conv2d( d[1], d[2], kernel_size=(3,3) )
        self.conv4 = nn.Conv2d( d[2], d[3], kernel_size=(3,3) )
        self.dense1 = nn.Linear( d[3], d[4] )
        self.dense2 = nn.Linear( d[4], 10 )

        self.a = nn.ReLU()
        self.mp = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.a(self.conv1(x))
        x = self.mp(x)
        x = self.a(self.conv2(x))
        x = self.mp(x)
        x = self.a(self.conv3(x))
        x = self.a(self.conv4(x))
        x = self.a(self.dense1(x.view(batch_size, -1)))
        x = self.dense2(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)
        self.a = nn.ReLU()

    def forward(self, x):
        orig_size = x.size()
        x = x.view(x.size(0), -1)
        x = self.a(self.l1(x))
        x = self.a(self.l2(x))
        x = self.l3(x)
        return x

class FullyConv(nn.Module):
    def __init__(self):
        super(FullyConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3) )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3) )
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3) )
        self.conv4 = nn.Conv2d(64, 10, kernel_size=(3,3) )
        # self.a = nn.ReLU()
        self.a = F.relu
        self.mp = nn.MaxPool2d(kernel_size=2, stride=(2,2))

    def forward(self, x):
        x = self.a(self.conv1(x))
        x = self.mp(x)
        x = self.a(self.conv2(x))
        x = self.mp(x)
        x = self.a(self.conv3(x))
        x = self.conv4(x).squeeze()
        return x
