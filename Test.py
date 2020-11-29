from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 100
BATCH_SIZE = 50

def syncAgentsWeights(learningAgent, opponentAgent):
    opponentAgent.target_model.set_weights(learningAgent.target_model.get_weights())


if __name__ == '__main__':
    env = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learningAgent = MinMaxAgent(env, 3, 1)
    opponentAgent = MinMaxAgent(env, 1, 0)

    env.reset()
    env.render()
    while True:
        env.currently_playing_color = learningAgent.player_color
        tmp_state = deepcopy(env.state)
        learningAgent_action = learningAgent.best_action(tmp_state)
        env.step(learningAgent_action)
        env.render()
        env.currently_playing_color = opponentAgent.player_color
        tmp_state = deepcopy(env.state)
        opponentAgent_action = opponentAgent.best_action(tmp_state)
        env.step(opponentAgent_action)
        env.render()
        if env.done:
            break









