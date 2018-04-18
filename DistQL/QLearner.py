from Learner import Learner


class QLearner(Learner):
    def __init__(self):
        super(QLearner, self).__init__()

    def learn(self):
        pass

    def update(self, observation, previous_observation, reward):
        pass

    def select_action(self, observation, action_space):
        pass

