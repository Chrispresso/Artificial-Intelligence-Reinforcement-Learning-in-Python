import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

threshold = 1e-3
gamma = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all p(s',r|s,a) = 1 or 0

if __name__ == "__main__":
    grid = negative_grid()

    print('Rewards:')
    print_values(grid.rewards, grid)
    print('')

    policy = {}
    # Randomly choose action and update as we learn
    for state in grid.actions:
        policy[state] = np.random.choice(ALL_POSSIBLE_ACTIONS) 

    print('Initial Policy:')
    print_policy(policy, grid)

    V = {}
    states = grid.all_states()
    for state in states:
        # V(s) = 0
        if state in grid.actions:
            V[state] = np.random.random()
        # Terminal
        else:
            V[state] = 0

    while True:
        # Step 1. Policy Evaluation
        while True:
            biggest_change = 0
            for state in states:
                old_v = V[state]

                # V(s) only has a value if it is not a terminal state
                if state in policy:
                    action = policy[state]
                    grid.set_state(state)
                    reward = grid.move(action)
                    V[state] = reward + gamma * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[state]))

            if biggest_change < threshold:
                break
        
        # Step 2. Policy Improvement
        is_policy_converged = True
        for state in states:
            if state in policy:
                old_action = policy[state]
                new_action = None
                best_value = float('-inf')
                # Loop through all possible actions to find the best CURRENT action
                for action in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(state)  # Set current state for testing
                    reward = grid.move(action)
                    v = reward + gamma * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_action = action
                
                policy[state] = new_action  # Update our policy
                if new_action != old_action:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print('Values:')
    print_values(V, grid)
    print('Policy:')
    print_policy(policy, grid)