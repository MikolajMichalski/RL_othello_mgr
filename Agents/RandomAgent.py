
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Softmax
from copy import deepcopy
import sys
from Reversi_combined import ReversiEnv

class RandomAgent:

    def __init__(self, state_size, action_size, env, player_color):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer_size = 100
        self.memory = deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005
        self.env = env
        self.player_color = player_color
        self.agent_type = "random"

    def get_action_to_make(self, state):
            return random.choice(self.env.possible_actions)



