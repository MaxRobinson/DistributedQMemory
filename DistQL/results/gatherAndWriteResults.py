import json
import time
from threading import Thread

import requests

SERVER_URL = 'http://localhost:5000'


def get_and_save_data(num_agents, env_name, round):
    print('Saving Experiment Data for Agents:{}, env:{}, round{}'.format(num_agents, env_name, round))

    r = requests.get('http://localhost:5000/state')

    state = r.json()

    # write to file
    with open('result-update-{}-env-{}-agents-{}-round-{}.json'.format('Partial', env_name, num_agents, round), 'w') as f:
        json.dump(state, f, indent=4)

    return


def experiment(num_agents, env_name):

    for i in range(10):
        print('Start Experiment round: {}'.format(i))

        r = requests.post(SERVER_URL+"/experiment/start", json={"num_agents": num_agents, "env_name": env_name})

        time.sleep(10)
        # is_complete = False
        while True:
            r = requests.get(SERVER_URL+"/experiment/status")
            value = r.json()
            if value.get('complete', False):
                break
            else:
                time.sleep(10)

        get_and_save_data(num_agents, env_name, i)

        requests.get(SERVER_URL+"/state/clear")
        requests.get(SERVER_URL+"/q/clear")

    return


if __name__ == '__main__':
    experiment(8, 'Taxi-v2')

