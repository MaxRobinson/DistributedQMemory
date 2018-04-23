import json
import threading
import time

from flask import Flask, request

#### App Set up ###
from StateBuilder.StateBuilderCartPole import StateBuilderCartPole
from controller.Controller import Controller
from learner.QLearner import QLearner

app = Flask(__name__) # create the application instance :)


class QServer:
    ### STATICS
    ALPHA_INDEX = 1
    VALUE_INDEX = 0
    LEARNING_DECAY = .99


    Q = {}

    initial_alpha = .99


class StateBuilderCache:
    builders = {
        'CartPole-v1': StateBuilderCartPole(),
        'LunarLander-v2': None,
        'Taxi-v2': None
    }


class Results:
    pass


class State:
    num_agents = 1
    complete = False



@app.route('/update/q', methods=['POST'])
def hello_world():
    # not error checking right now for json
    body = request.get_json()

    update_q(body['q_updates'])

    return json.dumps(QServer.Q)


@app.route('/q', methods=['GET'])
def get_q():
    return json.dumps(QServer.Q)


@app.route('/experiment/start', methods=['POST'])
def start_experiment():
    body = request.get_json()

    num_agents = body.get('num_agents', 1)
    env_name = body.get('env_name', 'Taxi-v2')
    update_frequency = body.get('update_freq', 10)

    state_builder = StateBuilderCache.builders.get(env_name, None)

    start_workers(num_agents, env_name, state_builder, update_frequency)

    return 'Success'

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


def start_workers(num_agents, env_name, state_builder, update_frequency):

    for agent in range(num_agents):
        controller = Controller(None, 'Taxi-v2', None, id=agent)

        learner = QLearner(controller.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9, decay_rate=.999)

        controller.set_learner(learner)

        agent_thread = threading.Thread(target=controller.train,
                                        kwargs={"number_epochs": 2001, "save_location": '../models/{}-{}.model'.format(env_name, agent)})
        agent_thread.start()

    return


if __name__ == '__main__':
    app.run()

