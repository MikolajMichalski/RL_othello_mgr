
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

class HumanAgent:

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

    def get_action_to_make(self, state):
            self.env.render()
            possible_actions_to_make = ReversiEnv.get_possible_actions(self.env.state, self.player_color)

            possible_actions_to_make_coords = dict()
            i = 1

            for action in possible_actions_to_make:
                possible_actions_to_make_coords[i] = ReversiEnv.action_to_coordinate(self.env.state, action)
                i += 1

            print(f"Possible actions to make: \n")

            for key, value in possible_actions_to_make_coords.items():
                print(f"{key})  x:{value[1] + 1} y:{value[0] + 1} \n")

            valid_input = False
            while valid_input == False:
                human_input = input("Please chose action")

                try:
                    int_input = int(human_input)
                    if int_input not in possible_actions_to_make_coords.keys():
                        print("Please input proper possible action number!!!")
                        continue
                    valid_input = True
                except ValueError:
                    print("Incorrect input!!! Please input integer value!!!")
                    continue

            human_action = ReversiEnv.coordinate_to_action(self.env.state, possible_actions_to_make_coords[int_input])

            return human_action



