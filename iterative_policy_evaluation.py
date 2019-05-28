import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, Grid


threshold = 10e-4


def print_values(values, grid: Grid) -> None:
    for i in range(grid.width):
        print('------------------------')
        for j in range(grid.height):
            v = values.get((i, j), 0)
            if v >= 0:
                print(' %.2f|' % v, end='')
            else:
                print('%.2f|' % v, end='')
        print('')


def print_policy(policy, grid: Grid) -> None:
    for i in range(grid.width):
        print('------------------------')
        for j in range(grid.height):
            action = policy.get((i, j), ' ')
            print('  %s  |' % action, end='')
        print('')


if __name__ == '__main__':
    grid = standard_grid()
    states = grid.all_states()

    ## Uniformly random actions ###
    # Initialize V(s) = 0
    V = {}
    gamma = 1.0
    for state in states:
        V[state] = 0

    while True:
        biggest_change = 0
        for state in states:
            old_v = V[state]

            # V(s) only has a value if it is not a terminal state
            if state in grid.actions:
                new_v = 0
                prob_a = 1.0 / len(grid.actions[state])  # Equal prob

                for action in grid.actions[state]:
                    grid.set_state(state)
                    reward = grid.move(action)
                    new_v += prob_a * (reward + gamma * V[grid.current_state()])
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))

        if biggest_change < threshold:
            break

    print('Values for uniformly random actions:')
    print_values(V, grid)
    print('\n\n')

    ### Fixed Policy ###
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
    print('Fixed policy:')
    print_policy(policy, grid)
    print('')

    # Initialize V(s) = 0
    V = {}
    for state in states:
        V[state] = 0

    gamma = .9

    # Repeat until convergence
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
    print('Values for fixed policy:')
    print_values(V, grid)
    print('\n\n')
