import json
from load_data import load_data
import requests as req


def get_discounts(dataloader):
    model1_logs = open('model1.log', 'w')
    basic_model_logs = open('basic_model.log', 'w')

    for X, _ in dataloader:
        for x in X:
            x_ = x.squeeze().tolist()[:4]
            request = f'd_time={x_[0]}&n_visits={x_[1]}&visits_per_minute={x_[2]}&max_disc={x_[3]}'
            model1_logs.write(request)
            basic_model_logs.write(request)
            model1_request = 'http://127.0.0.1:5000/predict/model1?' + request
            basic_model_request = 'http://127.0.0.1:5000/predict/basic-model?' + request
            resp = req.get(model1_request)
            resp.raise_for_status()
            json = resp.json()
            model1_logs.write('&predicted_disc=' +
                              str(json['discount']) + '\n')
            resp = req.get(basic_model_request)
            resp.raise_for_status()
            json = resp.json()
            basic_model_logs.write(
                '&predicted_disc=' + str(json['discount']) + '\n')


if __name__ == '__main__':
    _, test_dataLoader = load_data(
        "data3/normal_vpm.json", batch_size=128, test_size=0.1)
    get_discounts(test_dataLoader)
