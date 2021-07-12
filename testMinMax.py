from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
from Agents.RandomAgent import RandomAgent
from collections import deque
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 1000
BATCH_SIZE = 30
outputFilePath = "Output/MinMax_Play.txt"
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

def start(number_of_layers, min_max_depth, verbose):

    envTest = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = envTest.observation_space.shape[0]
    action_size = envTest.action_space.n
    agent = DDQNAgent(state_size, envTest, 0, number_of_layers)


    weights_to_load = ""

    if number_of_layers == 2:
        weights_to_load = "Weights/2_layers_weights.h5"
    elif number_of_layers == 3:
        weights_to_load = "Weights/3_layers_weights.h5"
    elif number_of_layers == 4:
        weights_to_load = "Weights/4_layers_weights.h5"


    agent.load(weights_to_load)
    agent.epsilon = 0
    opponent = MinMaxAgent(envTest, min_max_depth, 1)
    games_won = 0
    games_lost = 0
    games_tied = 0
    test_win_percentage = 0.
    test_lost_percentage = 0.
    test_tied_percentage =0.
    stateTest = envTest.reset()
    test_episodes_counter = 0

    print(f"Rozpoczęto rozgrywkę {EPISODES} gier przeciwko {agent.__class__.__name__}.")
    print(f"MinMax depth is {opponent.max_depth}")
    while True:
        stateTest = np.reshape(stateTest, [1, state_size])
        envTest.currently_playing_color = agent.player_color
        envTest.possible_actions = ReversiEnv.get_possible_actions(envTest.state, envTest.currently_playing_color)
        if len(envTest.possible_actions) != 0:

            agentAction = agent.get_action_to_make(stateTest)
            next_state_test, reward1, done, _ = envTest.step(agentAction)
            next_state_test = np.reshape(next_state_test, [1, state_size])

            stateTest = next_state_test

        else:

            envTest.pass_place_counter += 1
            if envTest.pass_place_counter > 1:
                done = True
        if verbose:
            envTest.render()
        envTest.currently_playing_color = opponent.player_color

        envTest.possible_actions = ReversiEnv.get_possible_actions(envTest.state, envTest.currently_playing_color)

        envTestCopy = deepcopy(envTest)
        if len(envTest.possible_actions) != 0:
            opponentAgent_action = opponent.best_action(envTestCopy.state)
            next_state_test, reward2, done, _ = envTest.step(opponentAgent_action)
            next_state_test = np.reshape(next_state_test, [1, state_size])

        else:
            envTest.pass_place_counter += 1
            if envTest.pass_place_counter > 1:
                done = True

        stateTest = next_state_test
        if verbose:
            envTest.render()
        if done:
            test_episodes_counter += 1
            black_score_test = len(np.where(envTest.state[0, :, :] == 1)[0])
            white_score_test = len(np.where(envTest.state[1, :, :] == 1)[0])

            if black_score_test > white_score_test:
                games_won += 1
                test_win_percentage = games_won / test_episodes_counter * 100
                test_lost_percentage = games_lost / test_episodes_counter * 100
                test_tied_percentage = games_tied / test_episodes_counter * 100

                writeStdOutputToFile(outputFilePath,
                                     f"Wynik gry {test_episodes_counter} - C/B: {black_score_test}/{white_score_test}\n"
                                     f"Procent wyganych gier agenta RL: {test_win_percentage}\n"
                                     f"Procent przegranch gier agenta RL: {test_lost_percentage}\n"
                                     f"Procent zremisowanych gier agenta RL: {test_tied_percentage}\n")
            elif black_score_test < white_score_test:
                games_lost += 1
                test_win_percentage = games_won / test_episodes_counter * 100
                test_lost_percentage = games_lost / test_episodes_counter * 100
                test_tied_percentage = games_tied / test_episodes_counter * 100

                writeStdOutputToFile(outputFilePath,
                                     f"Wynik gry {test_episodes_counter} - C/B: {black_score_test}/{white_score_test}\n"
                                     f"Procent wyganych gier agenta RL: {test_win_percentage}\n"
                                     f"Procent przegranch gier agenta RL: {test_lost_percentage}\n"
                                     f"Procent zremisowanych gier agenta RL: {test_tied_percentage}\n")
            elif black_score_test == white_score_test:
                games_tied += 1
                test_win_percentage = games_won / test_episodes_counter * 100
                test_lost_percentage = games_lost / test_episodes_counter * 100
                test_tied_percentage = games_tied / test_episodes_counter * 100

                writeStdOutputToFile(outputFilePath,
                                     f"Wynik gry {test_episodes_counter} - C/B: {black_score_test}/{white_score_test}\n"
                                     f"Procent wyganych gier agenta RL: {test_win_percentage}\n"
                                     f"Procent przegranch gier agenta RL: {test_lost_percentage}\n"
                                     f"Procent zremisowanych gier agenta RL: {test_tied_percentage}\n")
            stateTest = envTest.reset()
        if test_episodes_counter == EPISODES:
            break











