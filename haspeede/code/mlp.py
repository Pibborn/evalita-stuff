import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import math
import numpy as np
from sklearn.utils import shuffle

def preprocess_data(X, y):
    '''
    Converts the input data (X, y) into a form compatible with PyTorch.
    Specifically, it converts X and y into FloatTensors from respecively
    a sparse scipy matrix and a list of strings.
    '''
    X = np.array(X.todense())
    X = torch.from_numpy(X).float()
    y = np.array(list(map(int, y)))
    y = torch.from_numpy(y).long()
    return X, y

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, size in enumerate(layer_sizes):
            if i == len(layer_sizes) - 1:
                break
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.ReLU())
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, X, y, X_test, y_test, batch_size = 32, lr = 1e-6, num_epochs = 100):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        X, y = shuffle(X, y, random_state=0)
        X, y = preprocess_data(X, y)
        X_test, y_test = preprocess_data(X_test, y_test)
        batches_per_epoch = math.ceil(len(X) / batch_size)
        for i in range(num_epochs):
            loss_acc = 0
            for j in range(batches_per_epoch):
                batch_x = X[batch_size * j: batch_size * (j+1)]
                batch_y = y[batch_size * j: batch_size * (j+1)]
                optimizer.zero_grad()
                y_pred = self(batch_x)
                loss_value = self.loss(y_pred, batch_y)
                loss_value.backward()
                loss_acc += loss_value
                optimizer.step()
            train_loss = loss_acc / batches_per_epoch
            y_pred = self(X_test)
            test_loss = self.loss(y_pred, y_test)
            y_pred = y_pred.max(1, keepdim=True)[1]
            accuracy = y_test.eq(y_pred.permute(1, 0)).sum().numpy() / y_test.shape[0]
            print('Epoch {}: train loss {} test loss {} test accuracy {}'.format(i, train_loss, test_loss, accuracy))


