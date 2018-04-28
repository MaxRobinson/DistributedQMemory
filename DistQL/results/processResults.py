import json

import numpy as np
import scipy.stats
import scipy as sp

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

def process_results(tau:int=10, DQL_type:str="ALL", env_name:str="Taxi-v2", legend_loc:int=0):
    plt.figure()
    for agents in [1, 2, 4, 8]:
        episode_matrix = np.zeros((10, 2001))
        for i in range(10):
            if tau == 10:
                filename = 'result-update-{}-env-{}-agents-{}-round-{}.json'.format(DQL_type, env_name, agents, i)
            else:
                filename = 'result-tau-{}-update-{}-env-{}-agents-{}-round-{}.json'.format(tau, DQL_type, env_name, agents, i)

            with open(filename, 'r') as f:

                state = json.load(f)

                # get average over all agents
                agent_scores = []
                for agent in range(agents):
                    reward_list = state['experimental_results']['results'][str(agent)]['cumulative_reward']
                    agent_scores.append(reward_list)
                agent_scores = np.array(agent_scores)
                agent_mean_scores = np.mean(agent_scores, axis=0)

                # x = np.array(reward_list)
                episode_matrix[i] = agent_mean_scores

        # avg_cumulative = episode_matrix.sum(axis=0)
        # avg_cumulative = avg_cumulative/len(episode_matrix)

        means = np.mean(episode_matrix, axis=0)

        standard_errors = scipy.stats.sem(episode_matrix, axis=0)
        # print(standard_errors)

        a = len(episode_matrix[0])
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


        if agents == 1:
            label = "{} agent".format(agents)
        else:
            label = "{} agents".format(agents)

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
    title_part2 = "with DistQL-{} tau {} in {}".format(DQL_type, tau, env_name)
    plt.title("\n".join([title, title_part2]))

    # plt.savefig('{}.svg'.format((title + title_part2).replace(' ', '-')))
    plt.show()
    plt.close()


if __name__ == '__main__':
    process_results(10, 'ALL', 'Taxi-v2', 4)
    process_results(10, 'Partial', 'Taxi-v2', 4)
    process_results(10, 'ALL', 'CartPole-v1', 4)
    process_results(10, 'Partial', 'CartPole-v1', 4)

    process_results(50, 'ALL', 'Taxi-v2', 4)
    process_results(100, 'ALL', 'Taxi-v2', 4)
    process_results(50, 'ALL', 'CartPole-v1', 4)
    process_results(100, 'ALL', 'CartPole-v1', 4)
