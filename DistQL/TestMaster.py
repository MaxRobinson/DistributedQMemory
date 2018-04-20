from Controller import Controller
from QLearner import QLearner
from StateBuilderCartPole import StateBuilderCartPole


def main():

    cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole())

    learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9)

    cart_pole_ctrl.set_learner(learner)

    count = 0
    while True:
        cart_pole_ctrl.train_single(render=False)
        count += 1
        print("Epoch {}".format(count))
        if count % 100 == 0:
            cart_pole_ctrl.save("models/cart-pole.model")


def use_model():
    cart_pole_ctrl = Controller(None, 'CartPole-v1', StateBuilderCartPole())

    learner = QLearner(cart_pole_ctrl.get_action_space(), epsilon=0.1, init_alpha=.5, gamma=.9)

    cart_pole_ctrl.set_learner(learner)
    cart_pole_ctrl.load("models/cart-pole.model")

    count = 0
    while True:
        cart_pole_ctrl.run(render=True)
        count += 1
        print("Epoch {}".format(count))


if __name__ == '__main__':
    # main()
    use_model()

