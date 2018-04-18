from copy import copy

from Learner import Learner
from RandomLearner import RandomLearner
import logging
import gym

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, learner: Learner = None, env_id: str=None):
        if learner is None:
            logger.error("No Learner provided, not initializing")
            return
        else:
            self.learner = learner

        if env_id is None:
            logger.error("No env_id provided")
            return

        try:
            self.env = gym.make(env_id)
        except Exception as ex:
            logging.error("Error creating Gym", ex)
            raise ex

    def train_single(self, render: bool=False):
        observation = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            previous_observation = copy(observation)
            action = self.learner.select_action(previous_observation, self.env.action_space)

            observation, reward, done, info = self.env.step(action)
            self.learner.update(observation, previous_observation, reward)


if __name__ == "__main__":
    ctrl = Controller(learner=RandomLearner(), env_id='LunarLander-v2')
    ctrl.train_single(render=True)


