import numpy
import pandas

from StateBuilder.StateBuilder import StateBuilder


class StateBuilderLunarLander(StateBuilder):

    def __init__(self):
        super().__init__()

        self.n_bins = 10
        self.n_bins_binary = 2

        # Number of states is huge so in order to simplify the situation
        # we discretize the space to: 10 ** number_of_features
        self.x_bins = pandas.cut([-.5, .5], bins=self.n_bins, retbins=True)[1][1:-1]
        # self.y_bins = pandas.cut([0, .5], bins=self.n_bins, retbins=True)[1][1:-1]
        self.y_bins = numpy.array([0.02, 0.05, .07, .1, .13, .15, .2, .3, .4, .5])

        self.f2_bins = pandas.cut([-.5, .5], bins=self.n_bins, retbins=True)[1][1:-1]
        self.f3_bins = pandas.cut([-.5, .5], bins=self.n_bins, retbins=True)[1][1:-1]
        self.f4_bins = pandas.cut([-.5, .5], bins=self.n_bins, retbins=True)[1][1:-1]
        self.f5_bins = pandas.cut([-.5, .5], bins=self.n_bins, retbins=True)[1][1:-1]

        self.f6_bins = pandas.cut([0, 1], bins=self.n_bins, retbins=True)[1][1:-1]
        self.f7_bins = pandas.cut([0, 1], bins=self.n_bins, retbins=True)[1][1:-1]

    def build_state(self, features):
        return "-".join(map(lambda feature: str(int(feature)), features))

    def to_bin(self, value, bins):
        return numpy.digitize(x=[value], bins=bins)[0]

    def build_state_from_obs(self, observation):

        x, y, f2, f3, f4, f5, f6, f7 = observation

        state = self.build_state([self.to_bin(x, self.x_bins),
                                  self.to_bin(y, self.y_bins),
                                  self.to_bin(f2, self.f2_bins),
                                  self.to_bin(f3, self.f3_bins),
                                  self.to_bin(f4, self.f4_bins),
                                  self.to_bin(f5, self.f5_bins),
                                  self.to_bin(f6, self.f6_bins),
                                  self.to_bin(f7, self.f7_bins)])

        return state



