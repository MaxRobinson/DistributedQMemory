from learner.Learner import Learner


class RandomLearner(Learner):
    def __init__(self):
        super(RandomLearner, self).__init__()

    def learn(self):
        pass

    def update(self, observation, previous_observation, reward):
        pass

    def select_action(self, observation, action_space):
        print([x for x in range(action_space.n)])
        return action_space.sample()
