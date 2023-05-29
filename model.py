import torch
from torch.nn import *

class SingleLayerPerceptron(torch.nn.Module):
    def __init__(self, width):
        # call the parent constructor
        super(SingleLayerPerceptron, self).__init__()
        self.width = width
        
        self.l1 = Linear(384 * width, 9)
        self.nl1 = ReLU()
    
    def forward(self, x):
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[2] * self.width)
        # print(x.shape)
        x = self.l1(x)
        # x = self.nl1(x)

        return x.view(x.shape[0], x.shape[-1])

class MultiLayerPerceptron2(torch.nn.Module):
    def __init__(self, width):
        # call the parent constructor
        super(MultiLayerPerceptron2, self).__init__()
        self.width = width

        self.nn = Sequential(
            Linear(384 * width, 384 * width // 16),
            ReLU(),
            Linear(384 * width // 16, 100),
            ReLU(),
            Linear(100, 9)
        )
    
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2] * self.width)
        x = self.nn(x)
        return x.view(x.shape[0], x.shape[-1])

class MultiLayerPerceptron3(torch.nn.Module):
    def __init__(self, width):
        # call the parent constructor
        super(MultiLayerPerceptron3, self).__init__()
        self.width = width

        self.nn = Sequential(
            Linear(384 * width, 384 * width // 8),
            ReLU(),
            Linear(384 * width // 8, 384 * width // 16),
            ReLU(),
            Linear(384 * width // 16, 100),
            ReLU(),
            Linear(100, 9)
        )
    
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2] * self.width)
        x = self.nn(x)
        return x.view(x.shape[0], x.shape[-1])

class MultiLayerPerceptron4(torch.nn.Module):
    def __init__(self, width):
        # call the parent constructor
        super(MultiLayerPerceptron4, self).__init__()
        self.width = width

        self.nn = Sequential(
            Linear(384 * width, 384 * width // 4),
            ReLU(),
            Linear(384 * width // 4, 384 * width // 8),
            ReLU(),
            Linear(384 * width // 8, 384 * width // 16),
            ReLU(),
            Linear(384 * width // 16, 100),
            ReLU(),
            Linear(100, 9)
        )
    
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2] * self.width)
        x = self.nn(x)
        return x.view(x.shape[0], x.shape[-1])
