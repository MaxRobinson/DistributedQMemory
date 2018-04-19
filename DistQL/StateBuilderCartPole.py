from abc import ABCMeta

import numpy
import pandas

from StateBuilder import StateBuilder


class StateBuilderCartPole(StateBuilder):

    def __init__(self):
        super().__init__()

        self.n_bins = 8
        self.n_bins_angle = 10

        # Number of states is huge so in order to simplify the situation
        # we discretize the space to: 10 ** number_of_features
        self.cart_position_bins = pandas.cut([-2.4, 2.4], bins=self.n_bins, retbins=True)[1][1:-1]
        self.pole_angle_bins = pandas.cut([-2, 2], bins=self.n_bins, retbins=True)[1][1:-1]
        self.cart_velocity_bins = pandas.cut([-1, 1], bins=self.n_bins, retbins=True)[1][1:-1]
        self.angle_rate_bins = pandas.cut([-3.5, 3.5], bins=self.n_bins_angle, retbins=True)[1][1:-1]

    def build_state(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def to_bin(self, value, bins):
        return numpy.digitize(x=[value], bins=bins)[0]

    def build_state_from_obs(self, observation):

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation

        state = self.build_state([self.to_bin(cart_position, self.cart_position_bins),
                                  self.to_bin(pole_angle, self.pole_angle_bins),
                                  self.to_bin(cart_velocity, self.cart_velocity_bins),
                                  self.to_bin(angle_rate_of_change, self.angle_rate_bins)])

        return state



