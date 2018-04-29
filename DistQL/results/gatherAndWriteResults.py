import json
import time
from threading import Thread

import requests

SERVER_URL = 'http://localhost:5000'


def get_and_save_data(num_agents, env_name, round, update_freq, DQL_TYPE:str='ALL'):
    print('Saving Experiment Data for Agents:{}, env:{}, round: {}, tau:{}'.format(num_agents, env_name, round, update_freq))

    r = requests.get('http://localhost:5000/state')

    state = r.json()

    # write to file
    with open('result-tau-{}-update-{}-env-{}-agents-{}-round-{}.json'.format(update_freq, DQL_TYPE, env_name, num_agents, round), 'w') as f:
        json.dump(state, f, indent=4)

    return


def experiment(num_agents, env_name, update_frequency, dql_type):
    requests.get(SERVER_URL+"/state/clear")
    requests.get(SERVER_URL+"/q/clear")

    for i in range(10):
        print('Start Experiment round: {}'.format(i))

        r = requests.post(SERVER_URL+"/experiment/start", json={"num_agents": num_agents,
                                                                "env_name": env_name,
                                                                "update_freq": update_frequency})

        time.sleep(3)
        # is_complete = False
        while True:
            r = requests.get(SERVER_URL+"/experiment/status")
            value = r.json()
            if value.get('complete', False):
                break
            else:
                time.sleep(1)

        get_and_save_data(num_agents, env_name, i, update_frequency, dql_type)

        requests.get(SERVER_URL+"/state/clear")
        requests.get(SERVER_URL+"/q/clear")

    return


def set_dql_type(dql_type:str='ALL'):
    if dql_type == 'Partial':
        r = requests.post(SERVER_URL+"/DQL/all", json={"DQL_ALL": False})
    else:
        r = requests.post(SERVER_URL+"/DQL/all", json={"DQL_ALL": True})

    print(r.json())

    return

if __name__ == '__main__':
    set_dql_type('ALL')
    experiment(1, 'Taxi-v2', 10, 'ALL')
    experiment(2, 'Taxi-v2', 10, 'ALL')
    experiment(4, 'Taxi-v2', 10, 'ALL')
    experiment(8, 'Taxi-v2', 10, 'ALL')

    set_dql_type('Partial')
    experiment(1, 'Taxi-v2', 10, 'Partial')
    experiment(2, 'Taxi-v2', 10, 'Partial')
    experiment(4, 'Taxi-v2', 10, 'Partial')
    experiment(8, 'Taxi-v2', 10, 'Partial')

    set_dql_type('ALL')
    experiment(1, 'Taxi-v2', 50, 'ALL')
    experiment(2, 'Taxi-v2', 50, 'ALL')
    experiment(4, 'Taxi-v2', 50, 'ALL')
    experiment(8, 'Taxi-v2', 50, 'ALL')

    experiment(1, 'Taxi-v2', 100, 'ALL')
    experiment(2, 'Taxi-v2', 100, 'ALL')
    experiment(4, 'Taxi-v2', 100, 'ALL')
    experiment(8, 'Taxi-v2', 100, 'ALL')

    set_dql_type('Partial')
    experiment(1, 'Taxi-v2', 50, 'Partial')
    experiment(2, 'Taxi-v2', 50, 'Partial')
    experiment(4, 'Taxi-v2', 50, 'Partial')
    experiment(8, 'Taxi-v2', 50, 'Partial')

    experiment(1, 'Taxi-v2', 100, 'Partial')
    experiment(2, 'Taxi-v2', 100, 'Partial')
    experiment(4, 'Taxi-v2', 100, 'Partial')
    experiment(8, 'Taxi-v2', 100, 'Partial')

    # CartPole with adjusted Tau
    # experiment(1, 'CartPole-v1', 10)
    # experiment(2, 'CartPole-v1', 10)
    # experiment(4, 'CartPole-v1', 10)
    # experiment(8, 'CartPole-v1', 10)

    set_dql_type('ALL')
    experiment(1, 'CartPole-v1', 50, 'ALL')
    experiment(2, 'CartPole-v1', 50, 'ALL')
    experiment(4, 'CartPole-v1', 50, 'ALL')
    experiment(8, 'CartPole-v1', 50, 'ALL')

    experiment(1, 'CartPole-v1', 100, 'ALL')
    experiment(2, 'CartPole-v1', 100, 'ALL')
    experiment(4, 'CartPole-v1', 100, 'ALL')
    experiment(8, 'CartPole-v1', 100, 'ALL')

    set_dql_type('Partial')
    experiment(1, 'CartPole-v1', 50, 'Partial')
    experiment(2, 'CartPole-v1', 50, 'Partial')
    experiment(4, 'CartPole-v1', 50, 'Partial')
    experiment(8, 'CartPole-v1', 50, 'Partial')

    experiment(1, 'CartPole-v1', 100, 'Partial')
    experiment(2, 'CartPole-v1', 100, 'Partial')
    experiment(4, 'CartPole-v1', 100, 'Partial')
    experiment(8, 'CartPole-v1', 100, 'Partial')

