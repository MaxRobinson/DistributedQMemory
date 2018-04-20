from abc import ABCMeta
from abc import abstractmethod


class Learner(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def select_action(self, observation, action_space):
        pass

    @abstractmethod
    def update(self, observation, previous_observation, action_taken, reward):
        pass

    @abstractmethod
    def save(self, location):
        pass

    @abstractmethod
    def load(self, location):
        pass
