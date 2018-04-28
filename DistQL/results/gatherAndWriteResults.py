import json
import time
from threading import Thread

import requests

SERVER_URL = 'http://localhost:5000'


def get_and_save_data(num_agents, env_name, round, update_freq):
    print('Saving Experiment Data for Agents:{}, env:{}, round: {}, tau:{}'.format(num_agents, env_name, round, update_freq))

    r = requests.get('http://localhost:5000/state')

    state = r.json()

    # write to file
    with open('result-tau-{}-update-{}-env-{}-agents-{}-round-{}.json'.format(update_freq, 'Partial', env_name, num_agents, round), 'w') as f:
        json.dump(state, f, indent=4)

    return


def experiment(num_agents, env_name, update_frequency):

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

        get_and_save_data(num_agents, env_name, i, update_frequency)

        requests.get(SERVER_URL+"/state/clear")
        requests.get(SERVER_URL+"/q/clear")

    return


if __name__ == '__main__':
    experiment(1, 'Taxi-v2', 50)
    experiment(2, 'Taxi-v2', 50)
    experiment(4, 'Taxi-v2', 50)
    experiment(8, 'Taxi-v2', 50)

    experiment(1, 'Taxi-v2', 100)
    experiment(2, 'Taxi-v2', 100)
    experiment(4, 'Taxi-v2', 100)
    experiment(8, 'Taxi-v2', 100)

    # CartPole with adjusted Tau
    experiment(1, 'CartPole-v1', 50)
    experiment(2, 'CartPole-v1', 50)
    experiment(4, 'CartPole-v1', 50)
    experiment(8, 'CartPole-v1', 50)

    experiment(1, 'CartPole-v1', 100)
    experiment(2, 'CartPole-v1', 100)
    experiment(4, 'CartPole-v1', 100)
    experiment(8, 'CartPole-v1', 100)
