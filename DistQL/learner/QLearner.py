import pickle
import random
import sys
from typing import Tuple, Set

from learner.Learner import Learner


class QLearner(Learner):
    QVALUE_INDEX = 0
    ALPHA_INDEX = 1

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

        self.last_updated_states = set()

    def learn(self):
        pass

    def decay(self):
        self.epsilon *= self.decay_rate

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
            self.q[previous_observation][action_taken] = [reward, self.init_alpha]

        current_q_value = self.q[previous_observation][action_taken][QLearner.QVALUE_INDEX]

        alpha = self.q[previous_observation][action_taken][QLearner.ALPHA_INDEX]
        self.q[previous_observation][action_taken][QLearner.QVALUE_INDEX] = \
            current_q_value + alpha * (reward + (self.gamma * self.get_max_value(observation, self.q)) - current_q_value)

        self.decay_alpha(previous_observation, action_taken)

        self.last_updated_states.add((previous_observation, action_taken))

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

    # <editor-fold desc="Helpers">
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
            if q[current_state][action][QLearner.QVALUE_INDEX] > max_value:
                max_arg = action
                max_value = q[current_state][action][QLearner.QVALUE_INDEX]

        return max_arg

    def get_max_value(self, state, q):
        """
        returns the max value for a state action pair.
        """
        max_value = -sys.maxsize

        if state not in q:
            return 0
        for action in q[state]:
            if q[state][action][QLearner.QVALUE_INDEX] > max_value:
                max_value = q[state][action][QLearner.QVALUE_INDEX]

        return max_value

    def decay_alpha(self, state, action):
        self.q[state][action][QLearner.ALPHA_INDEX] *= self.decay_rate

    # </editor-fold>

    def get_value(self, state, action) -> tuple:
        q_and_alpha = [None, None]
        try:
            q_and_alpha = self.q[state][action]
        except Exception as e:
            print("Exception getting value for Q table")
            print(e)
            q_and_alpha = [None, None]

        return tuple(q_and_alpha)

    def set_value(self, state, action, value, alpha):
        try:
            if state not in self.q:
                self.q[state] = {}

            if action not in self.q[state]:
                # If we've never seen this state before, set the value an alpha
                self.q[state][action] = [value, alpha]
            else:
                # if we've seen this state and action only update the value, not the alpha
                self.q[state][action][QLearner.QVALUE_INDEX] = value
        except Exception as e:
            print("Exception Setting Value for Q table")
            print(e)
        return

    def get_last_updated_states(self) -> Set:
        return self.last_updated_states

    def reset_last_updated_states(self):
        self.last_updated_states = set()

    def save(self, location):
        """
        Saves the Q state
        """
        # output = open('q_save_file_normal_large_world{}.txt'.format(count), 'wb')
        with open(location, 'wb') as output:
            pickle.dump(self.q, output)

    def load(self, location):
        """
        Saves the Q state
        """
        with open(location, 'rb') as output:
            self.q = pickle.load(output)

    def set_model(self, model):
        self.q = model

