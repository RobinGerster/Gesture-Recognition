from dataloader import EmbeddingsDataloader
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from model import *
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import time
import numpy as np
import copy
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def run_training(model, learning_rate, batch_size, frames_per_datapoint,\
                 train_data_overlap, epochs):
    
    model = model(frames_per_datapoint).to(device)
    # print(model)
    best_model = copy.deepcopy(model.state_dict())
    train_dataset = EmbeddingsDataloader(width=frames_per_datapoint)
    test_dataset = EmbeddingsDataloader(width=frames_per_datapoint, mode='test', overlap=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=512)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def test():
        with torch.no_grad():
            total_attempts = 0
            correct = 0
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                res = model(x)
                # label = one_hot(y[:,-1].type(torch.int64), num_classes=9).float()
                total_attempts += x.shape[0]
                correct += float((y[:,-1] == torch.argmax(res, dim=-1)).sum())
            return round(correct / total_attempts, 3)


    steps = 0
    losses = []
    test_accs = []
    loss_count = 0
    for epoch in range(epochs):
        total_attempts = 0
        correct = 0
        index = 0
        test_acc_last = 0
        best_test_acc = 0
        for i, (x, y) in enumerate(train_dataloader):

            # get the samples
            x, y = x.to(device), y.to(device)

            # run through the model
            optimizer.zero_grad()
            res = model(x)

            # perform gradient descent
            label = one_hot(y[:,-1].type(torch.int64), num_classes=9).float()
            # print(res.shape, label.shape)
            loss = criterion(res, label)
            loss.backward()
            optimizer.step()

            # run test

            # update metrics
            loss_count = loss.item()
            steps += 1
            losses.append(loss.item())
            total_attempts += x.shape[0]
            corr = (y[:,-1] == torch.argmax(res, dim=-1)).sum()
            correct += corr
            if i % (len(train_dataloader) // 20) == 0:
                test_acc = test()
                test_accs.append(test_acc)
                test_acc_last = test_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_model = copy.deepcopy(model.state_dict())
            else:
                test_accs.append(test_acc_last)
            print(f"epoch {epoch}/{epochs-1}, done fraction: {round(index / len(train_dataloader), 2)}, loss is {round(loss_count, 3)}, accuracy {round((correct / total_attempts).item(), 3)}, test_acc: {test_acc}", end='\r')
            index += 1
            # time.sleep(0.01)
    return losses, test_accs, best_model

class Run:
    def __init__(self, model=SingleLayerPerceptron, learning_rate=0.01, batch_size=16,\
                 frames_per_datapoint=16, train_data_overlap=False, epochs=2):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.frames_per_datapoint = frames_per_datapoint
        self.train_data_overlap = train_data_overlap
        self.epochs = epochs
    
    def run_experiment(self):
        losses, test_accs, best_model = run_training(self.model, self.learning_rate, self.batch_size,\
                     self.frames_per_datapoint, self.train_data_overlap, self.epochs)
        self.losses, self.test_accs, self.best_model = losses, test_accs, best_model
        return losses, test_accs, best_model
    
    def plot_results(self):
        nrm_losses = (np.array(self.losses) / max(self.losses)) * max(self.test_accs)
        plt.plot(range(len(self.losses)), self.test_accs, label='test accuracies')
        plt.plot(range(len(self.losses)), nrm_losses, label='losses')
        plt.legend()
        plt.show()