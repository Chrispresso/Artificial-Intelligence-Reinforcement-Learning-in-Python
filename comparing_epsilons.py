import numpy as np
import matplotlib.pyplot as plt


class Bandit(object):
    def __init__(self, true_mean):
        self.m = true_mean
        self.mean = 0  # Our estimate of the Bandit's mean
        self.N = 0
    
    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - (1.0 / self.N))*self.mean + ((1.0 / self.N) * x)


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for i in range(N):
        # Epsilon Greedy
        p = np.random.random()
        if p < eps:
            selection = np.random.choice(3)
        else:
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
    print('Epsilon =', eps)
    print('Estimate of mean    Actual mean')
    for bandit in bandits:
        print('{:<20}{}'.format(bandit.mean, bandit.m))

    return cumulative_avg


if __name__ == '__main__':
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
