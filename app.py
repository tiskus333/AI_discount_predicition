from flask import Flask, jsonify, request

#import pandas as pd 
#import numpy as np 
#import seaborn as sns
#import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import TensorDataset, DataLoader
#from tqdm import tqdm

class Model1 (nn.Module):
    def __init__(self):
        super(Model1,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5,20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(20,10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(10,1),
            nn.Tanh()
        )
    def forward(self,x):
        return self.model(x)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    pass
    if request.method == 'POST':
        model = Model1()
        model.load_state_dict(torch.load("model"))
        inp = [[0.100068132,2,0.0001321401,0,x*5] for x in range(5)]
        x = model(torch.Tensor(inp))
        return jsonify(x.squeeze().tolist())
    else:
        model = Model1()
        model.load_state_dict(torch.load("model"))
        inp = [[0.100068132,2,0.0001321401,0,x*5] for x in range(5)]
        x = model(torch.Tensor(inp))
        return jsonify(x.squeeze().tolist())

@app.route('/')
def index():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True) 