import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import run_experiment as run_experiment_eps
from optimistic_initial_values import run_experiment as run_experiment_opt


class Bandit(object):
    def __init__(self, true_mean):
        self.m = true_mean
        self.mean = true_mean  # Our estimate of the Bandit's mean
        self.N = 0
    
    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - (1.0 / self.N))*self.mean + ((1.0 / self.N) * x)


def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):
        # UCB1
        selection = np.argmax([bandit.mean + np.sqrt(2*np.log(i+1) / max(bandit.N, 1e-9)) for bandit in bandits])
            
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
    

if __name__ == '__main__':
    m1, m2, m3 = 1.0, 2.0, 3.0
    eps_1 = run_experiment_eps(m1, m2, m3, .1, 100000)
    ucb = run_experiment(m1, m2, m3, 100000)

    # Log scale plot
    plt.plot(eps_1, label='eps = .1')
    plt.plot(ucb, label='UCB')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # Linear plot
    plt.plot(eps_1, label='eps = .1')
    plt.plot(ucb, label='UCB')
    plt.legend()
    plt.show()
