import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

threshold = 1e-3
gamma = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# Add some randomness

if __name__ == "__main__":
    grid = negative_grid(step_cost=-1.0)

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
                new_v = 0
                if state in policy:
                    for action in ALL_POSSIBLE_ACTIONS:
                        if action == policy[state]:
                            prob = .5
                        else:
                            prob = .5/3
                        grid.set_state(state)
                        reward = grid.move(action)
                        new_v += prob * (reward + gamma * V[grid.current_state()])
                    V[state] = new_v
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
                    v = 0
                    for action2 in ALL_POSSIBLE_ACTIONS:
                        if action == action2:
                            prob = .5
                        else:
                            prob = .5/3
                        grid.set_state(state)
                        reward = grid.move(action2)
                        v += prob * (reward + gamma * V[grid.current_state()])
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
