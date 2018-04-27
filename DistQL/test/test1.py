import gym

from StateBuilder.StateBuilderLunarLander import StateBuilderLunarLander

env = gym.make('FrozenLake8x8-v0')
env.reset()

state_builder = StateBuilderLunarLander()

for i_episode in range(100):
    observation = env.reset()
    for t in range(1000):
        # env.render()

        print(observation)
        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        print(observation)
        # print(info)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("Done")







