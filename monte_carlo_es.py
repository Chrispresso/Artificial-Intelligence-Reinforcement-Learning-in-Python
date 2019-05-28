import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, Grid
from iterative_policy_evaluation import print_values, print_policy


gamma = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# Monte Carlo Exploring-Starts for finding optimal policy

def play_game(grid: Grid, policy):
    # Reset game to start at a random position.
    # Need to do this becasue of the current deterministic policy
    # we would never end up at certain states
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    state = grid.current_state()
    action = np.random.choice(ALL_POSSIBLE_ACTIONS)  # First action is uniformly random

    states_actions_rewards = [(state, action, 0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0

    while True:
        reward = grid.move(action)
        num_steps += 1
        state = grid.current_state()

        if state in seen_states:
            r = -10. / num_steps
            # Hack so we don't end up in an infinitely long episode bumping into a wall
            states_actions_rewards.append((state, None, r))
            break
        elif grid.is_game_over():
            states_actions_rewards.append((state, None, reward))
            break
        else:
            action = policy[state]
            states_actions_rewards.append((state, action, reward))
        seen_states.add(state)
            
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


def max_dict(d: dict):
    """
    Returns the argmax (key) and max (value) from a dictionary
    """
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


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
    for t in range(2000):
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

        # Update policy
        for state in policy:
            policy[state] = max_dict(Q[state])[0]

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