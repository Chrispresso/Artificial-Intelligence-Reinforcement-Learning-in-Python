import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import run_experiment as run_experiment_eps


class Bandit(object):
    def __init__(self, true_mean, upper_limit):
        self.m = true_mean
        self.mean = upper_limit  # Our estimate of the Bandit's mean
        self.N = 1

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - (1.0 / self.N))*self.mean + ((1.0 / self.N) * x)


def run_experiment(m1, m2, m3, N, upper_limit=10):
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]

    data = np.empty(N)

    for i in range(N):
        # optimistic intial values and a greedy approach
        selection = np.argmax([bandit.mean for bandit in bandits])
            
        # Pull from bandit and update sample mean
        x = bandits[selection].pull()
        bandits[selection].update(x)

        data[i] = x

    cumulative_avg = np.cumsum(data) / (np.arange(N) + 1)

    plt.plot(cumulative_avg)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()

    # Print our esimate of each bandits mean and their actual mean
    print('Estimate of mean    Actual mean')
    for bandit in bandits:
        print('{:<20}{}'.format(bandit.mean, bandit.m))

    return cumulative_avg


def compare_epsilon_greedy():
    m1, m2, m3 = 1.0, 2.0, 3.0
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)

    # Log scale plot
    plt.plot(c_1, label='eps = .1')
    plt.plot(c_05, label='eps = .05')
    plt.plot(c_01, label='eps = .01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Linear plot
    plt.plot(c_1, label='eps = .1')
    plt.plot(c_05, label='eps = .05')
    plt.plot(c_01, label='eps = .01')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    m1, m2, m3 = 1.0, 2.0, 3.0
    upper_limit = 10
    eps_1 = run_experiment_eps(m1, m2, m3, 0.1, 100000)
    eps_01 = run_experiment_eps(m1, m2, m3, 0.01, 100000)
    opt = run_experiment(m1, m2, m3, 100000, upper_limit)

    # Log scale plot
    plt.plot(eps_1, label='eps = .1')
    plt.plot(eps_01, label='eps = .01')
    plt.plot(opt, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Linear plot
    plt.plot(eps_1, label='eps = .1')
    plt.plot(eps_01, label='eps = .01')
    plt.plot(opt, label='optimistic')
    plt.legend()
    plt.show()
