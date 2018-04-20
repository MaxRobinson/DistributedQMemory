from copy import copy

from Learner import Learner
from StateBuilder import StateBuilder
from RandomLearner import RandomLearner
import logging
import gym

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, learner: Learner = None, env_id: str=None, state_builder: StateBuilder=None):
        # if learner is None:
        #     logger.error("No Learner provided, not initializing")
        #     return
        # else:
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

    def train_single(self, render: bool=False):
        observation = self.env.reset()
        done = False

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

            # discritize state
            if self.state_builder is not None:
                observation_disc = self.state_builder.build_state_from_obs(observation)
            else:
                observation_disc = copy(observation)

            self.learner.update(observation_disc, previous_observation, action, reward)

    def run(self, render: bool=False):
        observation = self.env.reset()
        done = False

        # self.learner.set_model(model)

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

            # discritize state
            # if self.state_builder is not None:
            #     observation_disc = self.state_builder.build_state_from_obs(observation)
            # else:
            #     observation_disc = copy(observation)

            # self.learner.update(observation_disc, previous_observation, action, reward)


    def get_action_space(self):
        return range(self.env.action_space.n)

    def set_learner(self, learner):
        self.learner = learner

    def save(self, file_location):
        self.learner.save(file_location)

    def load(self, file_location):
        self.learner.load(file_location)

if __name__ == "__main__":
    ctrl = Controller(learner=RandomLearner(), env_id='LunarLander-v2')
    ctrl.train_single(render=True)


