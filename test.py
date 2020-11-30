from main import get_data, BostonModel
import numpy as np
import torch

torch.manual_seed(20201129)

def t_get_data():
    X_train, X_test, Y_train, Y_test = get_data()
    print(f'X_train shape: {X_train.shape}')

def t_get_model():
    nn_model = BostonModel
    if nn_model:
        print('Model load: SUCCESS')
    else:
        print('Model load: FAIL')

def run():
    t_get_data()
    t_get_model()

run()
