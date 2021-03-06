"""
Game of Reversi
"""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding
from copy import deepcopy
from collections import deque
from termcolor import colored
import collections

def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            return state.shape[-1]**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy

class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi","human"]}

    def __init__(self, opponent, observation_type, illegal_place_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }

        self.currently_playing_color = 0

        self.pass_place_counter = 0
        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        self.state = np.zeros((3, self.board_size, self.board_size))
        centerL = int(self.board_size/2-1)
        centerR = int(self.board_size/2)
        self.state[2, :, :] = 1.0
        self.state[2, (centerL):(centerR+1), (centerL):(centerR+1)] = 0
        self.state[0, centerR, centerL] = 1
        self.state[0, centerL, centerR] = 1
        self.state[1, centerL, centerL] = 1
        self.state[1, centerR, centerR] = 1
        self.to_play = self.currently_playing_color
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
        self.done = False
        self.pass_place_counter = 0

        obs = self.getCurrentObservations(self.state)
        return obs

    def step(self, action):
        if self.done:
            obs = self.getCurrentObservations(self.state)
            return obs, 0., True, {'state': self.state}
        if len(self.possible_actions) == 0:
            self.pass_place_counter += 1
            if self.pass_place_counter <= 1:
                pass
            else:
                self.done = True
                obs = self.getCurrentObservations(self.state)

                black_player_score = len(np.where(self.state[0, :, :] == 1)[0])
                white_opponent_score = len(np.where(self.state[1, :, :] == 1)[0])

                if black_player_score > white_opponent_score:
                    reward = 10.
                else:
                    reward = -10.

                return obs, reward, self.done, {'state': self.state}

        elif ReversiEnv.resign_place(self.board_size, action):
            obs = self.getCurrentObservations(self.state)
            return obs, -10., True, {'state': self.state}
        elif not ReversiEnv.valid_place(self.state, action, self.currently_playing_color):
            if self.illegal_place_mode == 'raise':
                raise
            elif self.illegal_place_mode == 'lose':

                self.done = True
                obs = self.getCurrentObservations(self.state)
                return obs, -10., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
        else:
            ReversiEnv.make_place(self.state, action, self.currently_playing_color)

        reward = 0


        obs = self.getCurrentObservations(self.state)

        self.done = self.isFinished(self.state)

        if self.done:

            black_player_score = len(np.where(self.state[0, :, :] == 1)[0])
            white_opponent_score = len(np.where(self.state[1, :, :] == 1)[0])

            if black_player_score > white_opponent_score:
                reward = 10.
            else:
                reward = -10.

            return obs, reward, self.done, {'state': self.state}

        return obs, reward, self.done, {'state': self.state}

    def render(self, mode='human',  close=False):
        if close:
            return

        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(colored(' ' * 7))
        for j in range(board.shape[1]):
            outfile.write(colored(' ' +  str(j + 1) + '  | ', 'grey'))
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write(colored('-' * (board.shape[1] * 6 - 1), "grey"))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(colored(' ' +  str(i + 1) + '  |', "grey"))
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write(colored('  O  ', "blue"))
                elif board[0, i, j] == 1:
                    outfile.write(colored('  B  ', "green"))
                else:
                    outfile.write(colored('  W  ', "red"))
                outfile.write(colored('|', "grey"))
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write(colored('-' * (board.shape[1] * 7 - 1), "grey"))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def get_possible_actions(board, player_color):
        actions=[]
        d = board.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):
            for pos_y in range(d):
                if (board[2, pos_x, pos_y]==0):
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if(dx == 0 and dy == 0):
                            continue
                        nx = pos_x + dx
                        ny = pos_y + dy
                        n = 0
                        if (nx not in range(d) or ny not in range(d)):
                            continue
                        while(board[opponent_color, nx, ny] == 1):
                            tmp_nx = nx + dx
                            tmp_ny = ny + dy
                            if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                                break
                            n += 1
                            nx += dx
                            ny += dy
                        if(n > 0 and board[player_color, nx, ny] == 1):
                            actions.append(pos_x * d + pos_y)
        return actions

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        '''
        check whether there is any reversible places
        '''
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        if board[2, coords[0], coords[1]] == 1:
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while(board[opponent_color, nx, ny] == 1):
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    def isFinished(self, board):
        for x in range(board.shape[-1]):
            for y in range(board.shape[-1]):
                if board[2, x, y] == 1:
                    return False
        return True

    @staticmethod
    def getCurrentObservations(state):
        obs = np.empty([state.shape[-1] * 3, state.shape[-1] * 3])
        obs= np.concatenate(state, axis=None)
        return obs








