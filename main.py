from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from typing import Tuple

torch.manual_seed(20201129)

def get_data():
    #load data
    boston = load_boston()
    data = boston.data
    target = boston.target
    features = boston.feature_names

    #scale data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    #format data
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=20201129)
    Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test) 
    Y_train, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_test)

    return X_train, X_test, Y_train, Y_test

def permute_data(X: torch.Tensor, Y: torch.Tensor, seed=1) -> Tuple[torch.Tensor]:
    perm = torch.randperm(X.shape[0])
    return X[perm], Y[perm]

class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()

class BostonModel(PyTorchModel):
    def __init__(self, n_hidden=13, dropout=1.0):
        super().__init__()
        self.f1 = torch.nn.Linear(13, n_hidden)
        self.f2 = torch.nn.Linear(n_hidden, 1)
        if dropout < 1.0:
            self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 13

        x = self.f1(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = torch.sigmoid(x)
        x = self.f2(x)
        return x

class PyTorchTrainer():
    def __init__(self, model: PyTorchModel, optim: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss):
        self.model = model
        self.optim = optim
        self.loss = criterion
        self._check_optim_net_aligned()

    def _check_optim_net_aligned(self):
        assert self.optim.param_groups[0]['params'] == list(self.model.parameters()) 

    def _generate_batches(self, X: torch.Tensor, Y: torch.Tensor, size=32) -> Tuple[torch.Tensor]:
        N = X.shape[0]
        for i in range(0, N, size):
            X_batch, Y_batch = X[i:i+size], Y[i:i+size]
            yield X_batch, Y_batch

    def fit(self, X_train: torch.tensor, Y_train: torch.Tensor,
            X_test: torch.Tensor, Y_test: torch.Tensor,
            epochs=100, eval_every=10, batch_size=32):

        for e in range(epochs):
            X_train, Y_train = permute_data(X_train, Y_train)
            batch_generator = self._generate_batches(X_train, Y_train, batch_size)

            for i, (X_batch, Y_batch) in enumerate(batch_generator):
                self.optim.zero_grad() #reset gradients
                output = self.model(X_batch)
                loss = self.loss(output, Y_batch)
                loss.backward() #back-propagation
                self.optim.step() #update parameters

            if (e+1) % eval_every == 0:
                output = self.model(X_test)
                loss = self.loss(output, Y_test)
                print(e+1, loss)

def main():
    X_train, X_test, Y_train, Y_test = get_data()
    nn_model = BostonModel(dropout=0.1)
    #optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.25)
    criterion = torch.nn.MSELoss()
    trainer = PyTorchTrainer(nn_model, optimizer, criterion)
    trainer.fit(X_train, Y_train, X_test, Y_test, epochs=1000, eval_every=100)

if __name__ == '__main__':
    main()
