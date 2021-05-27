from flask import Flask, jsonify, request
import torch
from model import Model
#import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt


#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import TensorDataset, DataLoader
#from tqdm import tqdm

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    model = Model()
    model.load_state_dict(torch.load("model2020210527-224155"))

    d_time = request.form.get('d_time', type=float)
    n_visits = request.form.get('n_visits', type=int)
    visits_per_minute = request.form.get('visits_per_minute', type=float)
    max_disc = request.form.get('max_disc', type=int)

    print(d_time, n_visits, visits_per_minute, max_disc)
    inp = [[d_time, n_visits, visits_per_minute, max_disc, x*5]
           for x in range(5)]

    x = model(torch.Tensor(inp))
    return jsonify(x.squeeze().tolist())


@app.route('/')
def index():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
