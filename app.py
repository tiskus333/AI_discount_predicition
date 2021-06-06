from flask import Flask, jsonify, request
import torch
from model import Model

app = Flask(__name__)


@app.route('/predict/model1', methods=['GET'])
def predict_model_1():
    model = Model()
    model.load_state_dict(torch.load(
        '/home/tisek/sem6/AI_discount_predicition/model20210606-004012'))

    d_time = request.args.get('d_time', type=float)
    n_visits = request.args.get('n_visits', type=float)
    visits_per_minute = request.args.get('visits_per_minute', type=float)
    max_disc = request.args.get('max_disc', type=float)

    # print(request.args)
    inp = [[d_time, n_visits, visits_per_minute, max_disc, x*5]
           for x in range(5)]

    x = model(torch.Tensor(inp))
    # print(x)
    return jsonify(discount=best_discount2(x.squeeze().tolist(), 0.8))


def best_discount2(discount_list, threshold):
    for i, disc in enumerate(discount_list):
        if disc > threshold:
            return i * 5
    return i * 5


@app.route('/predict/basic-model', methods=['GET'])
def predict_basic_model():
    d_time = request.args.get('d_time', type=float)
    n_visits = request.args.get('n_visits', type=float)
    visits_per_minute = request.args.get('visits_per_minute', type=float)
    max_disc = request.args.get('max_disc', type=float)

    return jsonify(discount=20)


@ app.route('/')
def index():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
