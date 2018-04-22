from abc import ABCMeta, abstractmethod

import numpy


class StateBuilder(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    @abstractmethod
    def to_bin(self, value, bins):
        return numpy.digitize(x=[value], bins=bins)[0]

    @abstractmethod
    def build_state_from_obs(self, observation):
        pass
