from controller.Controller import Controller
from learner.QLearner import QLearner
from StateBuilder.StateBuilderCartPole import StateBuilderCartPole
from StateBuilder.StateBuilderLunarLander import StateBuilderLunarLander

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt


def main():
    # Taxi-v2
    cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole(), communicate=False)
    # cart_pole_ctrl = Controller(None, 'Taxi-v2', None, communicate=False)
    # cart_pole_ctrl = Controller(None, 'LunarLander-v2', state_builder=StateBuilderLunarLander(), communicate=False)
    # cart_pole_ctrl = Controller(None, 'FrozenLake-v0', None, communicate=False)

    running_cumulative_reward = []
    for _ in range(3):
        learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9, decay_rate=.999)
        cart_pole_ctrl.set_learner(learner)

        cumulative_reward, num_steps = cart_pole_ctrl.train(number_epochs=2001)
        running_cumulative_reward.append(cumulative_reward)



    ar = np.array(running_cumulative_reward)
    means = np.mean(ar, axis=0)

    standard_errors = scipy.stats.sem(ar, axis=0)
    uperconf = means + standard_errors
    lowerconf = means - standard_errors
    # avg_cumulative = ar.sum(axis=0)
    # avg_cumulative = avg_cumulative/len(running_cumulative_reward)

    x = np.arange(0, len(means))
    # plt.plot(x, means, 'o')

    z = np.polyfit(x, means, 5)
    p = np.poly1d(z)
    plt.plot(x, p(x))

    plt.fill_between(x, uperconf, lowerconf, alpha=0.3, antialiased=True)

    # plt.ylim(ymax=50, ymin=-800)

    plt.show()
    plt.close()

    # z = np.arange(0, len(num_steps))
    # plt.plot(z, num_steps)
    # plt.show()
    # plt.close()

    cart_pole_ctrl.env.close()



def use_model():
    cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole(), communicate=False)
    # cart_pole_ctrl = Controller(None, 'Taxi-v2', None)
    # cart_pole_ctrl = Controller(None, 'LunarLander-v2', state_builder=StateBuilderLunarLander(), communicate=False)

    learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.0, init_alpha=.5, gamma=.9)

    cart_pole_ctrl.set_learner(learner)
    cart_pole_ctrl.load("models/CartPole-v1-7.model")

    count = 0
    while True:
        cart_pole_ctrl.run(render=True)
        count += 1
        print("Epoch {}".format(count))


if __name__ == '__main__':
    # main()
    use_model()

