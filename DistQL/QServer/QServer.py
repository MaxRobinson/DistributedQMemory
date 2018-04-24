import json
import threading
import time

from flask import Flask, request

#### App Set up ###
from StateBuilder.StateBuilder import StateBuilder
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
    def __init__(self, agent_id: int=1, num_epochs: int=1, cumulative_reward: list=[], num_steps: list=[]):
        self.agent_id = agent_id
        self.num_epochs = num_epochs
        self.cumulative_reward = cumulative_reward
        self.num_steps = num_steps


class ExperimentResult:
    def __init__(self, num_agents: int=1):
        self.num_agents = num_agents
        # agent_id : results
        self.results = {}


class State:
    num_agents = 1
    num_finished_agents = 0
    num_updates = 0
    complete = False

    experimental_results = ExperimentResult()

    @staticmethod
    def get_values():
        return {
            'num_agents': State.num_agents,
            'num_finished_agents': State.num_finished_agents,
            'num_updates': State.num_updates,
            'complete': State.complete,
            'experimental_results': State.experimental_results.__dict__
        }

    @staticmethod
    def reset():
        State.num_agents = 1
        State.num_finished_agents = 0
        State.num_updates = 0
        State.complete = False


@app.route('/update/q', methods=['POST'])
def update_q_route():
    # not error checking right now for json
    body = request.get_json()
    update_q(body['q_updates'])

    State.num_updates += 1

    return json.dumps(QServer.Q)


@app.route('/experiment/start', methods=['POST'])
def start_experiment():
    body = request.get_json()

    num_agents = body.get('num_agents', 1)
    env_name = body.get('env_name', 'Taxi-v2')
    update_frequency = body.get('update_freq', 10)

    state_builder = StateBuilderCache.builders.get(env_name, None)

    # Modify Server State
    State.reset()
    State.num_agents = num_agents
    State.experimental_results = ExperimentResult(num_agents=num_agents)

    start_workers(num_agents, env_name, state_builder, update_frequency)

    return 'Success'


@app.route('/experiment/submit_results', methods=['POST'])
def submit_results():
    body = request.get_json()
    result = None
    try:
        agent_id = body.get('agent_id')
        num_epochs = body.get('num_epochs')
        cumulative_reward = body.get('cumulative_reward')
        num_steps = body.get('num_steps')

        result = Results(agent_id, num_epochs, cumulative_reward, num_steps)

        State.experimental_results.results[agent_id] = result
        State.num_finished_agents += 1

        if State.num_finished_agents == State.num_agents:
            State.complete = True

    except Exception as e:
        return 503

    return 'Success'

@app.route('/q', methods=['GET'])
def get_q():
    return json.dumps(QServer.Q)


@app.route('/q/clear', methods=['GET'])
def clear_q():
    QServer.Q.clear()
    return 'Success'


@app.route('/state', methods=['GET'])
def get_state():
    return json.dumps(State.get_values())

# <editor-fold desc="Helpers">
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


def start_workers(num_agents: int=1, env_name: str='', state_builder: StateBuilder=None, num_epochs: int= 2001,
                  update_frequency: int=10):

    State.num_agents = num_agents

    for agent in range(num_agents):
        controller = Controller(learner=None, env_id=env_name, state_builder=state_builder,
                                update_freq=update_frequency, id=agent)

        learner = QLearner(controller.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9, decay_rate=.999)

        controller.set_learner(learner)

        agent_thread = threading.Thread(target=controller.train,
                                        kwargs={"number_epochs": num_epochs, "save_location": '../models/{}-{}.model'.format(env_name, agent)})
        agent_thread.start()

    return
# </editor-fold>


if __name__ == '__main__':
    app.run()

