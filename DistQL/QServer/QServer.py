import json
import threading
import time

from flask import Flask, request

#### App Set up ###
app = Flask(__name__) # create the application instance :)


class QServer:
    ### STATICS
    ALPHA_INDEX = 1
    VALUE_INDEX = 0
    LEARNING_DECAY = .99


    Q = {}

    initial_alpha = .99


@app.route('/update/q', methods=['POST'])
def hello_world():
    # not error checking right now for json
    body = request.get_json()

    update_q(body['q_updates'])

    return json.dumps(QServer.Q)


@app.route('/q', methods=['GET'])
def get_q():
    return json.dumps(QServer.Q)


def update_q(states_to_update: list=None):

    # list of dicts
    # dicts are {'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}
    for update in states_to_update:
        s = update.get('state')
        a = update.get('action')
        update_value = float(update.get('value'))
        alpha_i = float(update.get('alpha'))

        if s not in QServer.Q:
            QServer.Q[s] = {}

        if a not in QServer.Q[s]:
            QServer.Q[s][a] = [0, QServer.initial_alpha]

        # perform actual update
        central_alpha = QServer.Q[s][a][QServer.ALPHA_INDEX]

        # make sure that our central learning rate is never higher than an agent's learning rate.
        if central_alpha > alpha_i:
            QServer.Q[s][a][QServer.ALPHA_INDEX] = alpha_i
            central_alpha = alpha_i

        learning_ratio = (central_alpha**2 / alpha_i)

        # update Central Q value
        QServer.Q[s][a][QServer.VALUE_INDEX] = (1 - learning_ratio) * QServer.Q[s][a][QServer.VALUE_INDEX] + \
                                               learning_ratio * update_value

        # update central learning rate (decay learning rate for that (s,a)
        QServer.Q[s][a][QServer.ALPHA_INDEX] = central_alpha * QServer.LEARNING_DECAY


if __name__ == '__main__':
    app.run()

