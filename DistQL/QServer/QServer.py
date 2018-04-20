import json
import threading
import time

from flask import Flask, request

#### App Set up ###
app = Flask(__name__) # create the application instance :)


### STATICS
ALPHA_INDEX = 1
VALUE_INDEX = 0
LEARNING_DECAY = .99


global Q
Q = {}

global initial_alpha
initial_alpha = .99



@app.route('/update/q', methods=['POST'])
def hello_world():
    # not error checking right now for json
    body = request.get_json()

    update_q(body['q_updates'])

    return json.dumps(Q)


def update_q(states_to_update: list=None):
    global Q
    # list of dicts
    # dicts are {'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}
    for update in states_to_update:
        s = update.get('state')
        a = update.get('action')
        update_value = float(update.get('value'))
        alpha_i = float(update.get('alpha'))

        if s not in Q:
            Q[s] = {}

        if a not in Q[s]:
            Q[s][a] = [0, initial_alpha]

        # perform actual update
        central_alpha = Q[s][a][ALPHA_INDEX]

        # make sure that our central learning rate is never higher than an agent's learning rate.
        if central_alpha > alpha_i:
            Q[s][a][ALPHA_INDEX] = alpha_i
            central_alpha = alpha_i

        learning_ratio = (central_alpha**2 / alpha_i)

        # update Central Q value
        Q[s][a][VALUE_INDEX] = (1 - learning_ratio) * Q[s][a][VALUE_INDEX] + learning_ratio * update_value

        # update central learning rate (decay learning rate for that (s,a)
        Q[s][a][ALPHA_INDEX] = central_alpha * LEARNING_DECAY


if __name__ == '__main__':
    app.run()

