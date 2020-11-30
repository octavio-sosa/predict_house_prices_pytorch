from main import get_data
import numpy as np
import torch

torch.manual_seed(20201129)

def t_get_data():
    X_train, X_test, Y_train, Y_test = get_data()
    print(f'X_train shape: {X_train.shape}')

def run():
    t_get_data()

run()
