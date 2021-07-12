from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
from Agents.RandomAgent import RandomAgent
from Agents.HumanAgent import HumanAgent
from collections import deque
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 100
BATCH_SIZE = 30
outputFilePath = "TestSave/Test_weights83.txt"
import sys

def syncAgentsWeights(learningAgent, opponentAgent):
    opponentAgent.target_model.set_weights(learningAgent.target_model.get_weights())

def writeStdOutputToFile(filePath, text):
    print(text)
    original_std_out = sys.stdout
    with open(filePath, "a") as f:
        sys.stdout = f
        print(text)
        sys.stdout = original_std_out

def start(number_of_layers, verbose):
    episodes_counter = 0
    last_ten_episodes_scores = deque(maxlen=10)
    env = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learningAgent = DDQNAgent(state_size, env, 0, number_of_layers)
    learningAgent.epsilon = 0.
    weights_to_load = ""

    if number_of_layers == 2:
        weights_to_load = "Weights/2_layers_weights.h5"
    elif number_of_layers == 3:
        weights_to_load = "Weights/3_layers_weights.h5"
    elif number_of_layers == 4:
        weights_to_load = "Weights/4_layers_weights.h5"


    learningAgent.load(weights_to_load)
    opponentAgent = HumanAgent(state_size, action_size, env, 1)
    games_won = 0
    games_lost = 0
    games_tied = 0
    best_win_percentage = 0.
    test_win_percentage = 0.
    test_lost_percentage = 0.
    test_tied_percentage = 0.
    state = env.reset()

    print(f"EXECUTING {EPISODES} TEST GAMES AGAINST {opponentAgent.__class__.__name__}.")


    while True:
        env.render()
        reward1 = 0
        reward2 = 0
        state1 = None
        learningAgent_action = None
        if episodes_counter == EPISODES:
            break
        state = np.reshape(state, [1, state_size])
        env.currently_playing_color = learningAgent.player_color

        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)
        if len(env.possible_actions) != 0:
            learningAgent_action = learningAgent.get_action_to_make(state)
            next_state, reward1, done, _ = env.step(learningAgent_action)
            next_state = np.reshape(next_state, [1, state_size])

            state1 = state
            state = next_state

        else:
            env.pass_place_counter += 1



        env.currently_playing_color = opponentAgent.player_color

        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)

        if len(env.possible_actions) != 0:
            opponentAgent_action = opponentAgent.get_action_to_make(state)
            next_state, reward2, done, _ = env.step(opponentAgent_action)
            next_state = np.reshape(next_state, [1, state_size])
        else:
            env.pass_place_counter += 1

        if learningAgent_action is not None:
            reward = reward1 - reward2
            learningAgent.replay_buffer_save(state1, learningAgent_action, reward, next_state, done)
            state = next_state
        if env.pass_place_counter > 1:
            done = True


        if done:
            episodes_counter += 1

            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])

            if black_score>white_score:

                last_ten_episodes_scores.append(1)
                games_won += 1
                test_win_percentage = games_won / episodes_counter * 100
                test_lost_percentage = games_lost / episodes_counter * 100
                test_tied_percentage = games_tied / episodes_counter * 100

                writeStdOutputToFile(outputFilePath,
                                     f"Game {episodes_counter} result - B/W: {black_score}/{white_score}\n"
                                     f" RL Agent won games percentage: {test_win_percentage}\n"
                                     f" RL Agent lost games percentage: {test_lost_percentage}\n"
                                     f" RL Agent tied games percentage: {test_tied_percentage}\n")

            elif black_score<white_score:
                games_lost += 1
                test_win_percentage = games_won / episodes_counter * 100
                test_lost_percentage = games_lost / episodes_counter * 100
                test_tied_percentage = games_tied / episodes_counter * 100
                last_ten_episodes_scores.append(0)

                writeStdOutputToFile(outputFilePath,
                                     f"Game {episodes_counter} result - B/W: {black_score}/{white_score}\n"
                                     f" RL Agent won games percentage: {test_win_percentage}\n"
                                     f" RL Agent lost games percentage: {test_lost_percentage}\n"
                                     f" RL Agent tied games percentage: {test_tied_percentage}\n")
            elif black_score == white_score:
                games_tied += 1
                test_win_percentage = games_won / episodes_counter * 100
                test_lost_percentage = games_lost / episodes_counter * 100
                test_tied_percentage = games_tied / episodes_counter * 100

                writeStdOutputToFile(outputFilePath,
                                     f"Game {episodes_counter} result - B/W: {black_score}/{white_score}\n"
                                     f" RL Agent won games percentage: {test_win_percentage}\n"
                                     f" RL Agent lost games percentage: {test_lost_percentage}\n"
                                     f" RL Agent tied games percentage: {test_tied_percentage}\n")
            state = env.reset()











