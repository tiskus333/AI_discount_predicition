from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange


class Model (nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 20),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(20, 10),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(10, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataLoader, optimizer, criterion, epochs):
        self.train()
        t = trange(epochs, desc="Losses", leave=True)
        for epoch in t:
            train_loss = []
            for x, Y in train_dataLoader:
                x = x.cpu()
                Y = Y.cpu()
                optimizer.zero_grad()
                out = self.forward(x)
                loss = criterion(out, Y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                t.set_description(
                    f"Loss: {np.average(train_loss):.3f}", refresh=True)
        print(f"Loss: {np.average(train_loss):.3f}")
        torch.save(self.state_dict(), "models/model" +
                   datetime.now().strftime("%Y%m%d-%H%M%S"))

    def test(self, test_dataLoader, treshold=0.0):
        correct = 0
        errors = []
        for x, Y in test_dataLoader:
            self.eval()
            x = x.cpu()
            Y = Y.cpu()
            out = self.forward(x)
            out.cpu()
            for i, o in enumerate(out):
                if o * Y[i] >= treshold:
                    correct += 1
                else:
                    errors.append([x[i], Y[i]])
        print(
            f'Corectly predicted: {correct} out of {len(test_dataLoader.dataset)}\nTest accuracy: {correct/len(test_dataLoader.dataset):.2%}')
