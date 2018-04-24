from copy import copy

import requests
import json

from learner.Learner import Learner
from StateBuilder.StateBuilder import StateBuilder
from learner.RandomLearner import RandomLearner
import logging
import gym
import numpy as np


logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, learner: Learner = None, env_id: str=None, state_builder: StateBuilder=None, update_freq: int=10,
                 id: int=1, communicate: bool=True):
        self.learner = learner

        if env_id is None:
            logger.error("No env_id provided")
            return

        try:
            self.env = gym.make(env_id)
        except Exception as ex:
            logging.error("Error creating Gym", ex)
            raise ex

        self.state_builder = state_builder
        self.update_freq = update_freq
        self.id = id
        self.communicate = communicate

    def train(self, number_epochs: int=100, save_location: str='', render: bool=False):
        cumulative_reward = []
        num_steps = []
        for _ in range(number_epochs):
            cumulative, steps = self.train_single(render)

            # learner decay values
            self.learner.decay()

            cumulative_reward.append(cumulative)
            num_steps.append(steps)

            # Save every 100
            print("Epoch: {}".format(_))
            if _ % 100 == 0:
                self.save(save_location)

            # Update Server based on Update_frequency
            if _ % self.update_freq == 0 and self.communicate:
                self.update_server()
                # Ensure we are resetting states after sending updates for each learner
                self.learner.reset_last_updated_states()

        if self.communicate:
            self.submit_results(cumulative_reward, num_steps, number_epochs)

        return cumulative_reward, num_steps

    def train_single(self, render: bool=False):
        observation = self.env.reset()
        done = False

        # used for evaluating a run
        cumulative_reward = 0

        # number of steps taken
        num_steps = 0

        while not done:
            if render:
                self.env.render()

            previous_observation = copy(observation)

            # discritize state
            if self.state_builder is not None:
                previous_observation = self.state_builder.build_state_from_obs(previous_observation)

            # select action
            action = self.learner.select_action(previous_observation, self.env.action_space)

            # Step
            observation, reward, done, info = self.env.step(action)

            # discritize state
            if self.state_builder is not None:
                observation_disc = self.state_builder.build_state_from_obs(observation)
            else:
                observation_disc = copy(observation)

            # if done:
            #     reward -= 200

            # update metrics
            cumulative_reward += reward
            num_steps += 1

            self.learner.update(observation_disc, previous_observation, action, reward)

        return cumulative_reward, num_steps

    def run(self, render: bool=False):
        observation = self.env.reset()
        done = False

        # used for evaluating a run
        cumulative_reward = 0

        # number of steps taken
        num_steps = 0

        while not done:
            if render:
                self.env.render()

            previous_observation = copy(observation)

            # discritize state
            if self.state_builder is not None:
                previous_observation = self.state_builder.build_state_from_obs(previous_observation)

            action = self.learner.select_action(previous_observation, self.env.action_space)

            # Step
            observation, reward, done, info = self.env.step(action)

            # update metrics
            cumulative_reward += reward
            num_steps += 1

        if self.communicate:
            self.submit_aggregated_run(cumulative_reward, num_steps)

    # <editor-fold desc="Server Communication">
    def update_server(self):
        # body['q_updates'] = [{'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}, ]
        body = {}

        # a list of (state, action) tuples
        states_to_send = self.learner.get_last_updated_states()

        q_updates = []
        for state, action in states_to_send:
            value, alpha = self.learner.get_value(state, action)

            if type(state) == np.int64:
                state = state.item()

            state_action_update = {
                "state": state,
                "action": action,
                "alpha": alpha,
                "value": value
            }

            q_updates.append(state_action_update)

        body["q_updates"] = q_updates

        r = requests.post('http://localhost:5000/update/q', json=body)

        self.update_q_from_server_response(r)

    def submit_results(self, cumulative_reward: list=[], num_steps: list=[], num_epochs: int=10):
        logger.info("Agent {}: Submitting Results".format(self.id))
        body = {
            'agent_id': self.id,
            'num_epochs': num_epochs,
            'cumulative_reward': cumulative_reward,
            'num_steps': num_steps
        }

        r = requests.post('http://localhost:5000/experiment/submit_results', json=body)
        print(r)

    def submit_aggregated_run(self, cumulative_reward: int=0, num_steps: int=0):
        logger.info("Submitting Reference Run Results")
        body = {
            'cumulative_reward': cumulative_reward,
            'num_steps': num_steps
        }

        r = requests.post('http://localhost:5000/reference/results', json=body)
        print(r)

    # </editor-fold>

    # <editor-fold desc="Helpers">
    def update_q_from_server_response(self, response):
        # body = {'updates': values}
        # states_to_update: list=None
        json_response = response.json()
        states_to_update = json_response.get('updates')
        # list of dicts
        # dicts are {'state':state, 'action'=action, 'alpha'=alpha, 'value'=value}
        for update in states_to_update:
            s = update.get('state')
            a = update.get('action')
            update_value = float(update.get('value'))
            alpha = float(update.get('alpha'))

            self.learner.set_value(s, a, update_value, alpha)

        pass

    def get_action_space(self):
        return range(self.env.action_space.n)

    def set_learner(self, learner):
        self.learner = learner

    def save(self, file_location: str=''):
        if file_location == '':
            return

        self.learner.save(file_location)

    def load(self, file_location):
        self.learner.load(file_location)
    # </editor-fold>

if __name__ == "__main__":
    ctrl = Controller(learner=RandomLearner(), env_id='LunarLander-v2')
    ctrl.train_single(render=True)


