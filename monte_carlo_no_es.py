import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, Grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_es import max_dict

gamma = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# Monte Carlo NO Exploring-Starts for finding optimal policy

def random_action(action, eps=.1):
    prob = np.random.random()
    if prob < (1 - eps):
        return action
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid: Grid, policy):
    state = (2, 0)
    grid.set_state(state)
    action = random_action(policy[state])

    states_actions_rewards = [(state, action, 0)]
    while True:
        reward = grid.move(action)
        state = grid.current_state()
        if grid.is_game_over():
            states_actions_rewards.append((state, None, reward))
            break
        else:
            action = random_action(policy[state])
            states_actions_rewards.append((state, action, reward))
            
    # Calculate returns, G, by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for state, action, reward in reversed(states_actions_rewards):
        # Value of terminal state is 0 so ignore it. Can also ignore last G
        if first:
            first = False
        else:
            states_actions_returns.append((state, action, G))
        G = reward + gamma*G
    
    states_actions_returns.reverse()  # Order of states visited, which was reverse
    return states_actions_returns


if __name__ == "__main__":
    grid = negative_grid(-.1)

    print('Rewards:')
    print_values(grid.rewards, grid)
    print('')

    policy = {}
    for state in grid.actions:
        policy[state] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # Initialize Q(s) and returns
    Q = {}
    returns = {}  # Dictionary of state -> list of returns we've received
    states = grid.all_states()
    for state in states:
        if state in grid.actions:
            Q[state] = {}
            for action in ALL_POSSIBLE_ACTIONS:
                Q[state][action] = 0
                returns[(state, action)] = []
        # Terminal state or state we can't get to
        else:
            pass

    deltas = []
    for t in range(5000):
        if t % 100 == 0:
            print(t)

        # Generate an episode using policy
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for state, action, G in states_actions_returns:
            # First-visit
            sa = (state, action)
            if sa not in seen_state_action_pairs:
                old_q = Q[state][action]
                returns[sa].append(G)
                Q[state][action] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[state][action]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # Update policy argmax
        for state in policy:
            action, _ = max_dict(Q[state])
            policy[state] = action

    plt.plot(deltas)
    plt.show()

    print('Final Policy:')
    print_policy(policy, grid)
    print('')

    V = {}
    for state, Qs in Q.items():
        V[state] = max_dict(Q[state])[1]

    print('Final Values:')
    print_values(V, grid)
    print('')
    print_policy(policy, grid)