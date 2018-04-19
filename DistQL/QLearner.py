import pickle
import random
import sys

from Learner import Learner


class QLearner(Learner):
    def __init__(self, actions, epsilon=.99, init_alpha=.99, gamma=.9, decay_rate=.99):
        super(QLearner, self).__init__()
        self.epsilon = epsilon
        self.init_alpha = init_alpha
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.q = {}
        self.actions = actions

        self.sates_updated_since_sync = []

        self.episode_count = 0


    def learn(self):
        pass

    def update(self, observation, previous_observation, action_taken, reward):

        """
        Update q based on:
        q[s,a] = Q[s,a] + alpha(r + gamma* max_args(Q[s'])

        The main update function for updating a given q value.
        """
        # previous_state_hash = previous_observation

        # if action_taken is None:
        #     action_taken = ("end")

        if previous_observation not in self.q:
            self.q[previous_observation] = {}

        if action_taken not in self.q[previous_observation]:
            self.q[previous_observation][action_taken] = reward

        current_q_value = self.q[previous_observation][action_taken]

        self.q[previous_observation][action_taken] = \
            current_q_value + self.init_alpha * (reward + (self.gamma * self.get_max_value(observation, self.q)) - current_q_value)

        return self.q

    def select_action(self, observation, action_space):
        """
        Selects an action according to an epsilon-greedy policy.
        """
        role = random.random()
        action = None

        if role < self.epsilon:
            action = self.select_random_move(self.actions)
        else:
            action = self.arg_max(observation, self.q)

        if action is None:
            action = self.select_random_move(self.actions)

        return action

    def select_random_move(self, possible_actions: tuple) -> tuple:
        """
        Helper function to select a random move.
        """
        role = random.randint(0, len(possible_actions)-1)
        return possible_actions[role]

    def arg_max(self, current_state, q):
        """
        Returns the argmax for the current state. the arg max is the action that maximizes Q(s,a)
        """
        max_arg = None
        max_value = -sys.maxsize

        if current_state not in q:
            return None

        for action in q[current_state]:
            if q[current_state][action] > max_value:
                max_arg = action
                max_value = q[current_state][action]

        return max_arg

    def get_max_value(self, state, q):
        """
        returns the max value for a state action pair.
        """
        max_value = -sys.maxsize

        if state not in q:
            return 0
        for action in q[state]:
            if q[state][action] > max_value:
                max_value = q[state][action]

        return max_value

    def save(self, count):
        """
        Saves the Q state
        """
        output = open('q_save_file_normal_large_world{}.txt'.format(count), 'wb')
        pickle.dump(self.q, output)
        output.close()

