import gym

env = gym.make('LunarLander-v2')
env.reset()


for i_episode in range(100):
    observation = env.reset()
    for t in range(1000):
        # env.render()

        print(observation)
        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        # print(info)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("Done")







