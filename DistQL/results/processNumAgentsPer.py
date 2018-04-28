import json

import numpy as np
import scipy.stats
import scipy as sp

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

def process_agent_results(tau:int=10, num_agents: int=2, DQL_type:str= "ALL", env_name:str= "Taxi-v2", legend_loc:int=0):
    plt.figure()

    # episode_matrix = np.zeros((10, 2001))
    agent_dict = {}
    for i in range(num_agents):
        agent_dict[i] = np.zeros((10, 2001))

    for i in range(10):
        # with open('result-update-{}-env-{}-agents-{}-round-{}.json'.format(DQL_type, env_name, num_agents, i), 'r') as f:
        filename = 'result-tau-{}-update-{}-env-{}-agents-{}-round-{}.json'.format(tau, DQL_type, env_name, num_agents, i)
        with open(filename, 'r') as f:
            state = json.load(f)

            for agent in range(num_agents):
                reward_list = state['experimental_results']['results'][str(agent)]['cumulative_reward']
                agent_dict[agent][i] = reward_list

            # agent_scores = np.array(agent_scores)
            # agent_mean_scores = np.mean(agent_scores, axis=0)

            # x = np.array(reward_list)
            # episode_matrix[i] = agent_mean_scores

    agent_means = {}
    agent_stderr = {}
    for agent, array_of_rewards in agent_dict.items():
        means = np.mean(array_of_rewards, axis=0)
        standard_errors = scipy.stats.sem(array_of_rewards, axis=0)

        agent_means[agent] = means
        agent_stderr[agent] = standard_errors

        # a = len(array_of_rewards[0])
        # conf = standard_errors * sp.stats.t._ppf((1+.95)/2., a-1)

        # print(conf)

        # uperconf = means + conf
        # lowerconf = means - conf
        uperconf = means + standard_errors
        lowerconf = means - standard_errors

        # print(*avg_cumulative.tolist(), sep="\n")

        x = np.arange(0, len(means))
        # plt.plot(x, means, 'o')

        z = np.polyfit(x, means, 5)
        p = np.poly1d(z)

        label = "agent {}".format(agent)

        plt.plot(x, p(x), label=label, alpha=0.7)

        plt.fill_between(x, uperconf, lowerconf, alpha=0.3, antialiased=True)

    if env_name == 'Taxi-v2':
        plt.ylim(ymax=50, ymin=-800)
    else:
        plt.ylim(ymax=250, ymin=-200)

    plt.legend(loc=legend_loc)
    plt.xlabel("Episode Number")
    plt.ylabel("Cumulative Reward")
    title = "Average Performance and Standard Error "
    title_part2 = "of each {} agent with DistQL-{} tau {} in {}".format(num_agents, DQL_type, tau, env_name)
    plt.title("\n".join([title, title_part2]))

    plt.savefig('{}.svg'.format((title+title_part2).replace(' ', '-')))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # process_agent_results(2, 'ALL', 'Taxi-v2')
    # process_agent_results(50, 4, 'ALL', 'Taxi-v2')
    # process_agent_results(100, 4, 'ALL', 'Taxi-v2')
    # process_agent_results(8, 'ALL', 'Taxi-v2', 4)
    #
    # process_agent_results(2, 'Partial', 'Taxi-v2')
    # process_agent_results(50, 4, 'Partial', 'Taxi-v2')
    # process_agent_results(8, 'Partial', 'Taxi-v2', 4)
    #
    # process_agent_results(2, 'ALL', 'CartPole-v1')
    process_agent_results(50, 4, 'ALL', 'CartPole-v1')
    process_agent_results(100, 4, 'ALL', 'CartPole-v1')
    process_agent_results(50, 8, 'ALL', 'CartPole-v1')
    process_agent_results(100, 8, 'ALL', 'CartPole-v1')
    #
    # process_agent_results(2, 'Partial', 'CartPole-v1')
    # process_agent_results(50, 4, 'Partial', 'CartPole-v1')
    # process_agent_results(8, 'Partial', 'CartPole-v1', 2)

