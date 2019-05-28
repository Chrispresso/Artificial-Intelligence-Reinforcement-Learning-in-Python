import numpy as np
from grid_world import standard_grid, negative_grid, Grid
from iterative_policy_evaluation import print_values, print_policy


threshold = 1e-3
gamma = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# This is only policy evaluation, not optimization
def play_game(grid: Grid, policy):
    # Reset game to start at a random position.
    # Need to do this becasue of the current deterministic policy
    # we would never end up at certain states
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    state = grid.current_state()
    states_and_rewards = [(state, 0)]  # List of tuples (state, reward)
    while not grid.is_game_over():
        action = policy[state]
        reward = grid.move(action)
        state = grid.current_state()
        states_and_rewards.append((state, reward))

    # Calculate returns, G, by working backwards from the terminal state
    G = 0
    states_and_returns = []
    first = True
    for state, reward in reversed(states_and_rewards):
        # Value of terminal state is 0 so ignore it. Can also ignore last G
        if first:
            first = False
        else:
            states_and_returns.append((state, G))
        G = reward + gamma*G
    
    states_and_returns.reverse()  # Order of states visited, which was reverse
    return states_and_returns


if __name__ == "__main__":
    grid = standard_grid()

    print('Rewards:')
    print_values(grid.rewards, grid)
    print('')

    # State -> Action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U'
    }

    # Initialize V(s) and returns
    V = {}
    returns = {}  # Dictionary of state -> list of returns we've received
    states = grid.all_states()
    for state in states:
        if state in grid.actions:
            returns[state] = []
        # Terminal state or state we can't get to
        else:
            V[state] = 0
    
    for t in range(100):
        # Generate an episode using policy
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for state, G in states_and_returns:
            # Check to see if we have already seen this state
            # called 'first-visit' MC policy evaluation
            if state not in seen_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                seen_states.add(state)

    print('Values:')
    print_values(V, grid)
    print('')
    print_policy(policy, grid)