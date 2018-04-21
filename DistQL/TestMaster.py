from Controller import Controller
from QLearner import QLearner
from StateBuilderCartPole import StateBuilderCartPole

import numpy as np

import matplotlib.pyplot as plt


def main():
    # Taxi-v2
    # cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole())
    cart_pole_ctrl = Controller(None, 'Taxi-v2', None)

    learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9, decay_rate=.999)

    cart_pole_ctrl.set_learner(learner)

    running_cumulative_reward = []
    for _ in range(5):
        cumulative_reward, num_steps = cart_pole_ctrl.train(number_epochs=2000, save_location='models/taxi-v3.model')
        running_cumulative_reward.append(cumulative_reward)

    ar = np.array(running_cumulative_reward)
    avg_cumulative = ar.sum(axis=0)
    avg_cumulative = avg_cumulative/len(running_cumulative_reward)

    x = np.arange(0, len(cumulative_reward))
    plt.plot(x, avg_cumulative)

    plt.show()
    plt.close()

    z = np.arange(0, len(num_steps))
    plt.plot(z, num_steps)
    plt.show()
    plt.close()

    cart_pole_ctrl.env.close()



def use_model():
    cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole())
    # cart_pole_ctrl = Controller(None, 'Taxi-v2', None)

    learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9)

    cart_pole_ctrl.set_learner(learner)
    cart_pole_ctrl.load("models/cart-pole.model")

    count = 0
    while True:
        cart_pole_ctrl.run(render=True)
        count += 1
        print("Epoch {}".format(count))


if __name__ == '__main__':
    main()
    # use_model()

