import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class EmbeddingsDataloader(torch.utils.data.Dataset):

    def __init__(self, mode='train', overlap=False, width=16):
        self.mode = mode
        self.n = width # num of frames in one datapoint
        self.overlap = overlap

        self.prefix = "./train"
        self.prefix_labels = "./train"
        # self.mode_prefix = mode + '/'
        self.mode_prefix = '/'
        self.embedings_fileregex = '*_embeddings.pt'
        self.labels_fileregex = '*.csv'

        self.labels = self._load_labels()
        self.embeddings = self._load_embeddings()

        # print("UNIQUE", np.unique(self.labels))
        # # UNIQUE [0. 1. 2. 3. 4. 5. 6. 7. 8.]

        # print("datas shape", self.labels.shape, self.embeddings.shape)
        ## torch.Size([72843]) torch.Size([72843, 384]) on train
    
    def _csv_load(self, path):
        data = []
        with open(path, 'r') as f:
            ll = f.readlines()
            [data.append(int(l)) for l in ll[0].split(',')]
        return np.array(data)

    def _load_labels(self):
        files = glob.glob(self.prefix_labels + self.mode_prefix + self.labels_fileregex)
        files.sort()
        labels = np.array([])
        for file in files:
            labels = np.concatenate([labels, self._csv_load(file)])
        return torch.from_numpy(labels).float()

    def _load_embeddings(self):
        files = glob.glob(self.prefix + self.mode_prefix + self.embedings_fileregex)
        files.sort()
        embeddings = torch.tensor([])
        for file in files:
            embed_tensor = torch.load(file)
            embeddings = torch.cat([embeddings, embed_tensor])
        return embeddings.float()

    def __len__(self):
        if self.overlap:
            return len(self.labels) - self.n
        return len(self.labels) // self.n

    def __getitem__(self, idx):
        n = self.n
        if self.overlap:
            return  self.labels[idx:idx+n], self.embeddings[idx:idx+n].clone().detach()
        return  self.labels[idx*n:idx*n+n], self.embeddings[idx*n:idx*n+n].clone().detach()




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
class LstmRNN(nn.Module):
    """
        Parameters: 
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        """
        torch.nn.LSTM() Parameters
        - input_size: The number of expected features in the input x
        - hidden_size: The number of features in the hidden state h
        - num_layers: 多层堆叠. 
            E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, 
            with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
        - bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        - batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). 
            Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
        - dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
            with dropout probability equal to dropout. Default: 0
        - bidirectional: If True, becomes a bidirectional LSTM. Default: False
        - proj_size: If > 0, will use LSTM with projections of corresponding size. Default: 0
        """
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = nn.functional.relu(x)
        x = x.view(s, b, -1)
        return x    
    
    
if __name__ == '__main__':

    input_size = 384
    hidden_size = 32
    num_layers = 1
    output_size = 1
    # seq_length = 16
    batch_size = 1

    loader = EmbeddingsDataloader()
    labels, data = loader.__getitem__(0)
    print(data.__len__())
    print(labels.__len__())

    print(type(data))
    print(data.size())
    # print(dataset.__getitem__(0).size())
    # print(dataset.__getitem__(100).size())
    
    
    # Create LSTM model instance
    model = LstmRNN(input_size, hidden_size, output_size,num_layers)
    
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    # Training loop
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        for(labels,inputs) in loader:
            if(inputs.size()[0] != 16):
                break
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = inputs.reshape(1,16,384)
            # Forward pass
            output = model(inputs)
            output = output.view(16)
            print(output.size())
            # Compute loss
            loss = criterion(output, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Print progress
            if (epoch+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.cpu().item()}')
