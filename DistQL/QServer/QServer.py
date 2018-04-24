import copy
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
    INITIAL_ALPHA = .99

    Q = {}

    # This is a server mode that changes central servers are sent back to the workers in the response.
    # If True, the entire QServer Q Values are sent to the workers
    # If False, only the states the server updated from that worker are sent back.
    DQL_ALL = False


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
        # agent_id : results.__dict__
        self.results = {}


class ReferenceResult:
    def __init__(self):
        self.num_epochs = 0
        self.cumulative_reward = []
        self.num_steps = []


class State:
    env_name = ''
    num_agents = 1
    num_finished_agents = 0
    num_updates = 0
    complete = False

    experimental_results = ExperimentResult()

    reference_results = ReferenceResult()

    @staticmethod
    def get_values():
        return {
            'num_agents': State.num_agents,
            'num_finished_agents': State.num_finished_agents,
            'num_updates': State.num_updates,
            'complete': State.complete,
            'experimental_results': State.experimental_results.__dict__,
            'reference_results': State.reference_results.__dict__
        }

    @staticmethod
    def reset():
        State.env_name = ''
        State.num_agents = 1
        State.num_finished_agents = 0
        State.num_updates = 0
        State.complete = False
        State.experimental_results = ExperimentResult()
        State.reference_results = ReferenceResult()


#### Routes ####
@app.route('/update/q', methods=['POST'])
def update_q_route():
    # not error checking right now for json
    body = request.get_json()
    update_q(body['q_updates'])

    State.num_updates += 1

    # Start A reference learner to track progress
    if State.num_updates % State.num_agents == 0:
        start_reference_aggregated_learner(env_name=State.env_name)

    values = []
    if QServer.DQL_ALL:
        values = get_all_central_q_values()
    else:
        values = get_central_q_values(body['q_updates'])

    body = {
        'updates': values
    }
    response = json.dumps(body)

    return response


@app.route('/experiment/start', methods=['POST'])
def start_experiment():
    body = request.get_json()

    num_agents = body.get('num_agents', 1)
    env_name = body.get('env_name', 'Taxi-v2')
    update_frequency = body.get('update_freq', 10)

    state_builder = StateBuilderCache.builders.get(env_name, None)

    # Modify Server State
    State.reset()
    State.env_name = env_name
    State.num_agents = num_agents
    State.experimental_results = ExperimentResult(num_agents=num_agents)

    start_workers(num_agents, env_name, state_builder, num_epochs=2001, update_frequency=update_frequency)

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

        State.experimental_results.results[agent_id] = result.__dict__
        State.num_finished_agents += 1

        if State.num_finished_agents == State.num_agents:
            State.complete = True
            # run after all workers have stopped.
            start_reference_aggregated_learner(env_name=State.env_name)


    except Exception as e:
        return 503

    return 'Success'


@app.route('/reference/results', methods=['POST'])
def submit_reference_result():
    body = request.get_json()

    cumulative_reward_for_run = body.get('cumulative_reward')
    num_steps_for_run = body.get('num_steps')

    # Update state that is tracking reference results
    State.reference_results.num_epochs += 1
    State.reference_results.cumulative_reward.append(cumulative_reward_for_run)
    State.reference_results.num_steps.append(num_steps_for_run)

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


@app.route('/state/clear', methods=['GET'])
def clear_state():
    State.reset()
    return 'Success'


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
            QServer.Q[s][a] = [0, QServer.INITIAL_ALPHA]

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


def get_all_central_q_values():
    values = []
    # list of dicts
    # dicts are {'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}
    for s in QServer.Q:
        for a in QServer.Q[s]:
            update = {
                'state': s,
                'action': a,
                'value': QServer.Q[s][a][QServer.VALUE_INDEX],
                'alpha': QServer.Q[s][a][QServer.ALPHA_INDEX]
            }
            values.append(update)

    return values


def get_central_q_values(states_to_update: list=None):
    values = []
    # list of dicts
    # dicts are {'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}
    for update in states_to_update:
        s = update.get('state')
        a = update.get('action')

        update = {
            'state': s,
            'action': a,
            'value': QServer.Q[s][a][QServer.VALUE_INDEX],
            'alpha': QServer.Q[s][a][QServer.ALPHA_INDEX]
        }
        values.append(update)

    return values


def start_workers(num_agents: int=1, env_name: str='', state_builder: StateBuilder=None, num_epochs: int= 2001,
                  update_frequency:int=10):

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


def start_reference_aggregated_learner(env_name: str=''):
    state_builder = StateBuilderCache.builders.get(env_name, None)

    controller = Controller(learner=None, env_id=env_name, state_builder=state_builder)

    learner = QLearner(controller.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9, decay_rate=.999)

    # SET MODEL with copy of Server Model
    learner.set_model(copy.deepcopy(QServer.Q))

    controller.set_learner(learner)

    agent_thread = threading.Thread(target=controller.run)
    agent_thread.start()
    print('Started Reference Learner')

    return
# </editor-fold>


if __name__ == '__main__':
    app.run()

