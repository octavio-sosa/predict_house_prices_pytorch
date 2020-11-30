from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import torch

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

def main():
    X_train, X_test, Y_train, Y_test = get_data()

if __name__ == '__main__':
    main()
