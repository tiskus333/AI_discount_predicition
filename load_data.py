import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(path: str, batch_size: int, test_size: int):
    df = pd.read_json(path, orient="records", lines=True)
    list_data = df.to_numpy()
    test = int(test_size*len(list_data))
    train = len(list_data) - test
    train_split = [train, test]

    data_x = torch.Tensor(list_data[:, :-1])
    data_Y = torch.Tensor(list_data[:, -1].reshape(-1, 1))
    data = TensorDataset(data_x, data_Y)
    train_data, test_dataset = torch.utils.data.random_split(data, train_split)

    train_dataLoader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataLoader, test_dataLoader
