import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

LENGTH = 3  # Height/Width of board


class Environment(object):
    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1  # Represents 'x' on the board
        self.o = 1  # Represents 'o' on the board
        self.winner = None
        self.ended = False  # Has the game ended
        self.num_states = 3**(LENGTH*LENGTH)  # Number of possible states for the board

    def is_empty(self, row: int, col: int) -> bool:
        """
        Determines whether a spot on the board is empty
        """
        return self.board[row, col] == 0

    def game_over(self, force_recalculate: bool = False) -> bool:
        """
        Returns True if game is over (player won or it's a draw)

        Sets self.winner and self.ended
        """
        if not force_recalculate and self.ended:
            return self.ended

        # Check rows
        for row in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[row].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # Check columns
        for column in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, column].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # Check diagonals
        for player in (self.x, self.o):
            # Top-left -> Bottom-right diag
            if self.board.trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True
            # Top-right -> Bottom-left diag
            if np.fliplr(self.board).trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True

        # Check draw
        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True
        
        # Game is not over if there are still 0's on board
        self.winner = None
        return False

    def get_state(self) -> int:
        """
        Returns the current state of the board, represented as an int from 0...{S}-1.
        Where {S} = set of all possible states. {S} = 3^(LENGTH*LENGTH) since each
        board space can be 0, 1 or -1 representing empty, 'o', or 'x', respectively.

        Equivalent of finding value of base-3 number
        """
        k = 0
        h = 0
        for row in range(LENGTH):
            for col in range(LENGTH):
                if self.board[row, col] == 0:
                    val = 0
                elif self.board[row, col] == self.x:
                    val = 1
                elif self.board[row, col] == self.o:
                    val = 2
                
                h += (3**k) * val  # (base^position) * value
                k += 1  # Move to next position

        return h

    def is_draw(self) -> bool:
        """
        True if game is a draw
        """
        return self.ended and self.winner is None

    def draw_board(self) -> None:
        """
        Draws the board
        i.e.

        -------------
        | x |   |   |
        -------------
        |   |   |   |
        -------------
        |   |   | o |
        -------------
        """
        for row in range(LENGTH):
            print('--------------')
            print('| ', end='')
            for col in range(LENGTH):
                if self.board[row, col] == self.x:
                    print(' x |', end='')
                elif self.board[row, col] == self.o:
                    print(' o |', end='')
                else:
                    print('   |', end='')
            print('')  # End of column
        print('--------------')  # End of rows

    def reward(self, symbol: int) -> bool:
        """
        Get the reward for a agent's symbol
        """
        if not self.game_over():
            return 0

        # Game is over, did we win?
        return 1 if self.winner == symbol else 0


class Agent(object):
    def __init__(self, epsilon: float, learning_rate: float, symbol: int, verbose: bool = False):
        self.epsilon = epsilon  # Probability of choosing random action instead of greedy
        self.learning_rate = learning_rate
        self.state_history = []
        self.V = None
        self.verbose = verbose
        self.symbol = symbol

    def reset_history(self) -> None:
        """
        Reset self.state_history
        """
        self.state_history = []

    def update_state_history(self, state: int) -> None:
        """
        Update the state history
        """
        self.state_history.append(state)

    def update(self, environment: Environment) -> None:
        """
        Backtrack over the states such that:
            V(prev_state) = V(prev_state) + learning_rate*(V(next_state) - V([prev_state]))
            where V(next_state) = reward if it's the most current state

        @Note:
            We only do this at the end of an episode (for tic-tac-toe)
        """
        reward = environment.reward(self.symbol)
        target = reward
        for prev_state in reversed(self.state_history):
            value = self.V[prev_state] + self.learning_rate*(target - self.V[prev_state])
            self.V[prev_state] = value
            target = value
        self.reset_history()
    
    def take_action(self, environment: Environment) -> None:
        # Epsilon-greedy
        r = np.random.rand()
        best_state = None
        if r < self.epsilon:
            # Random action
            if self.verbose:
                print('Taking a random action')

            possible_moves = []
            for row in range(LENGTH):
                for col in range(LENGTH):
                    if environment.is_empty(row, col):
                        possible_moves.append((row, col))
            
            selection = np.random.choice(len(possible_moves))
            next_move = possible_moves[selection]
        # Greedy
        else:
            pos2value = {}  # For verbose
            next_move = None
            best_value = -1
            for row in range(LENGTH):
                for col in range(LENGTH):
                    if environment.is_empty(row, col):
                        # What is the state if we made this move?
                        environment.board[row, col] = self.symbol
                        state = environment.get_state()
                        environment.board[row, col] = 0  # Change it back
                        pos2value[(row, col)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (row, col)

            # If verbose, print board with values
            if self.verbose:
                print("Taking a greedy action")
                for row in range(LENGTH):
                    print("------------------")
                    for col in range(LENGTH):
                        if environment.is_empty(row, col):
                            # Print the value
                            print(" %.2f|" % pos2value[(row, col)], end="")
                        else:
                            print("  ", end="")
                            if environment.board[row, col] == environment.x:
                                print("x  |", end="")
                            elif environment.board[row, col] == environment.o:
                                print("o  |", end="")
                            else:
                                print("   |", end="")
                    print("")
                print("------------------")
        
        # Make the move
        environment.board[next_move[0], next_move[1]] = self.symbol


class Human(object):
    def __init__(self, symbol: int):
        self.symbol = symbol

    def take_action(self, envrionment: Environment) -> None:
        while True:
            move = input('Enter corrdinate row,col for your next move (row,col=0..2): ')
            row, col = move.split(',')
            row = int(row)
            col = int(col)
            if envrionment.is_empty(row, col):
                envrionment.board[row, col] = self.symbol
                break

    def update(self, environment: Environment) -> None:
        pass

    def update_state_history(self, state: int) -> None:
        pass


def get_state_hash_and_winner(environment: Environment, row: int = 0, col: int = 0):
    results = []
    for v in (0, environment.x, environment.o):
        environment.board[row, col] = v
        # End of the column, so go to next row
        if col == LENGTH-1:
            # If we are also at the end of the rows then the board is full
            if row == LENGTH-1:
                state = environment.get_state()
                ended = environment.game_over(True)
                winner = environment.winner
                results.append((state, winner, ended))
            # Just move to next row
            else:
                results += get_state_hash_and_winner(environment, row + 1, 0)
        # Just go to next column
        else:
            results += get_state_hash_and_winner(environment, row, col + 1)

    return results


def init_x_vals(envrionment: Environment, state_winner_triples: Tuple[int, int, bool]) -> 'np.ndarray':
    """
    Initialize state values such that
    if x wins, V(s) = 1
    if x loses or it's a draw, V(s) = 0
    otherwise, V(s) = 0.5
    """
    V = np.zeros((envrionment.num_states, 1))
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == envrionment.x:
                v = 1
            else:
                v = 0
        else:
            v = .5
        V[state] = v

    return V


def init_o_vals(envrionment: Environment, state_winner_triples: Tuple[int, int, bool]) -> 'np.ndarray':
    """
    Initialize state values such that
    if o wins, V(s) = 1
    if o loses or it's a draw, V(s) = 0
    otherwise, V(s) = 0.5
    """
    V = np.zeros((envrionment.num_states, 1))
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == envrionment.o:
                v = 1
            else:
                v = 0
        else:
            v = .5
        V[state] = v

    return V


def play_game(p1, p2, environment: Environment, draw: bool = False) -> int:
    current_player = None
    while not environment.game_over():
        # Alternate players
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        # Draw board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                environment.draw_board()
            elif draw == 2 and current_player == p2:
                environment.draw_board()

        current_player.take_action(environment)

        # Update state history
        state = environment.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
  
    if draw:
        environment.draw_board()

    p1.update(environment)
    p2.update(environment)

    if isinstance(p1, Human):
        if environment.winner is not None:
            if environment.winner == p1.symbol:
                return 1  # Win
            else:
                return 2  # Lose
        else:
            return 3  # Draw

    elif isinstance(p2, Human):
        if environment.winner is not None:
            if environment.winner == p2.symbol:
                return 1  # Win
            else:
                return 2  # Lose
        else:
            return 3  # Draw
    
    else:
        return None
    

if __name__ == '__main__':
    # Train the agent
    eps = .1
    learning_rate = 0.5
    environment = Environment()

    p1 = Agent(eps, learning_rate, environment.x)
    p2 = Agent(eps, learning_rate, environment.o)

    state_winner_triples = get_state_hash_and_winner(environment)

    Vx = init_x_vals(environment, state_winner_triples)
    p1.V = Vx
    Vo = init_o_vals(environment, state_winner_triples)
    p2.V = Vo

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment())

    # Play human vs. agent
    human = Human(environment.o)
    first = True
    while True:
        p1.verbose = True
        print('\nrow=0, col=0 is top left\n')
        # First time, let the AI go first
        if first:
            print('AI goes first')
            win = play_game(p1, human, Environment(), draw=2)
            first = False
        else:
            if np.random.rand() < .5:
                print('AI goes first')
                win = play_game(p1, human, Environment(), draw=2)
            else:
                print('You go first')
                win = play_game(human, p1, Environment(), draw=1)

        if win == 1:
            print('You won!!')
        elif win == 2:
            print('Sorry, you lost... try again!')
        elif win == 3:
            print('TIE! Try again!')

        answer = input('Play again? [Y/n]: ')

        if answer and answer.lower()[0] == 'n':
            break
