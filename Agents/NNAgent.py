
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

class DDQNAgent:

    def __init__(self, state_size, action_size, env, player_color):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer_size = 5000
        self.memory = deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self.initiate_model()
        self.target_model = self.initiate_model()
        self.env = env
        self.sync_target_model()
        self.player_color = player_color

    def initiate_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        # optimizer=Adam(lr=self.learning_rate))
        return model

    def replay_buffer_save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action_to_make(self, state):
        if np.random.rand() <= self.epsilon:
            #return random.randrange(64)
            return random.choice(self.env.possible_actions)
            #return random.choice(range(env.action_space.n))
        act_values = self.model.predict(state)
        possible_act_values = np.zeros((1, len(act_values[0])), float)
        for index in range(len(act_values[0])):
            if index in self.env.possible_actions:
                possible_act_values[0][index] = act_values[0][index]
        possible_action = np.argmax(possible_act_values[0])
        return possible_action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def sync_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
