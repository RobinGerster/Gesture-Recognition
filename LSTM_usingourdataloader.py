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



# class EmbeddingsDataloader(torch.utils.data.Dataset):

#     def __init__(self, mode='train', overlap=False, width=16):
#         self.mode = mode
#         self.n = width # num of frames in one datapoint
#         self.overlap = overlap

#         self.prefix = "./data/frame_embeddings/"
#         self.prefix_labels = "./data/orig_data/"
#         # self.mode_prefix = mode + '/'
#         self.mode_prefix = '/'
#         self.embedings_fileregex = '*_embeddings.pt'
#         self.labels_fileregex = '*.csv'

#         self.labels = self._load_labels()
#         self.embeddings = self._load_embeddings()

#         # print("UNIQUE", np.unique(self.labels))
#         # # UNIQUE [0. 1. 2. 3. 4. 5. 6. 7. 8.]

#         # print("datas shape", self.labels.shape, self.embeddings.shape)
#         ## torch.Size([72843]) torch.Size([72843, 384]) on train
    
#     def _csv_load(self, path):
#         data = []
#         with open(path, 'r') as f:
#             ll = f.readlines()
#             [data.append(int(l)) for l in ll[0].split(',')]
#         return np.array(data)

#     def _load_labels(self):
#         files = glob.glob(self.prefix_labels + self.mode_prefix + self.labels_fileregex)
#         files.sort()
#         labels = np.array([])
#         for file in files:
#             labels = np.concatenate([labels, self._csv_load(file)])
#         return torch.from_numpy(labels).float()

#     def _load_embeddings(self):
#         files = glob.glob(self.prefix + self.mode_prefix + self.embedings_fileregex)
#         files.sort()
#         embeddings = torch.tensor([])
#         for file in files:
#             embed_tensor = torch.load(file)
#             embeddings = torch.cat([embeddings, embed_tensor])
#         return embeddings.float()

#     def __len__(self):
#         if self.overlap:
#             return len(self.labels) - self.n
#         return len(self.labels) // self.n

#     def __getitem__(self, idx):
#         n = self.n
#         if self.overlap:
#             return  self.labels[idx:idx+n], self.embeddings[idx:idx+n].clone().detach()
#         return  self.labels[idx*n:idx*n+n], self.embeddings[idx*n:idx*n+n].clone().detach()




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
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        # self.dropout = torch.nn.Dropout(p=0.2)
 
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        # x = self.dropout(x)
        x = self.forwardCalculation(x)
        return x

    
if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    input_size = 384
    hidden_size = 192
    num_layers = 3
    
    # 与类别数相同
    output_size = 9
    seq_length = 16
    batch_size = 32

    # loader = EmbeddingsDataloader()
    
    from torch.utils.data import DataLoader 
    from dataloader import EmbeddingsDataloader
    dataset = EmbeddingsDataloader(mode='train', width=seq_length, overlap=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
    # Create LSTM model instance
    model = LstmRNN(input_size, hidden_size, output_size,num_layers)
    
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    
    # Training loop
    num_epochs = 20
    model.train()

    # for testing
    test_dataset = EmbeddingsDataloader(width=seq_length, mode='test', overlap=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512)
    def test():
        with torch.no_grad():
            model.eval()
            total_attempts = 0
            correct = 0
            classified = []
            ground_truth = []
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                res = model(x)
                # label = one_hot(y[:,-1].type(torch.int64), num_classes=9).float()
                total_attempts += x.shape[0]
                correct += float((y[:,-1] == torch.argmax(res, dim=-1)).sum())

                ground_truth.append(torch.nn.functional.one_hot(y[:,-1].type(torch.int64), num_classes=9).float().cpu().numpy())
                classified.append(res.cpu().numpy())
            return round(correct / total_attempts, 3), classified, ground_truth

    from tqdm import tqdm

    testaccs = []
    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()
        for inputs, labels in tqdm(loader):
            optimizer.zero_grad()
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            
            # Forward pass
            output = model(inputs)
            # output = output.view(16)
            
            # print(output.shape, torch.nn.functional.one_hot(labels[:,-1].to(torch.int64) , num_classes=9).shape)

            # Compute loss
            loss = criterion(output, torch.nn.functional.one_hot(labels[:,-1].to(torch.int64), num_classes=9).float())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Print progress
            # if (epoch+1) % 1 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.cpu().item()}')
        
        testacc, y_hat, y = test()
        print("FINAL ACC", testacc)
        testaccs.append(testacc)
    
    import matplotlib.pyplot as plt
    plt.plot(range(len(testaccs)), testaccs)
    plt.savefig('LSTM_learning.jpg')

    # # Test the model
    # model.eval()
    # with torch.no_grad():
    #     seq_length = 10
    #     test_input_data = torch.randn(batch_size, seq_length, input_size)
    #     test_target_data = torch.randn(batch_size, output_size)
    #     test_output = model(test_input_data)
    #     test_loss = criterion(test_output, test_target_data)
    #     print(f'Test Loss: {test_loss.item()}')
