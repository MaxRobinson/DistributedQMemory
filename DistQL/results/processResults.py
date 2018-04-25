import json

import numpy as np
import scipy.stats
import scipy as sp

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

def process_results():
    plt.figure()
    for agents in [1, 2, 4, 8]:
        episode_matrix = np.zeros((10, 2001))
        for i in range(10):
            # with open('result-update-ALL-env-Taxi-v2-agents-{}-round-{}.json'.format(agents, i), 'r') as f:
            with open('result-update-Partial-env-Taxi-v2-agents-{}-round-{}.json'.format(agents, i), 'r') as f:
            # with open('result-update-Partial-env-Taxi-v2-agents-1-round-{}.json'.format(i), 'r') as f:
            # with open('result-update-Partial-env-Taxi-v2-agents-2-round-{}.json'.format(i), 'r') as f:
            # with open('result-update-Partial-env-Taxi-v2-agents-4-round-{}.json'.format(i), 'r') as f:
            # with open('result-update-Partial-env-Taxi-v2-agents-8-round-{}.json'.format(i), 'r') as f:

                state = json.load(f)
                reward_list = state['experimental_results']['results']['0']['cumulative_reward']

                x = np.array(reward_list)

                episode_matrix[i] = x

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

        plt.plot(x, p(x), label=str(agents))

        plt.fill_between(x, uperconf, lowerconf, alpha=0.3, antialiased=True)

    plt.ylim(ymax=50, ymin=-800)

    plt.savefig('test.svg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    process_results()
