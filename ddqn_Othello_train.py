from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
from Agents.RandomAgent import RandomAgent
from Agents.nnAgentAdamOptim import DDQNAgentAdamOptim
from collections import deque
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 100000
BATCH_SIZE = 32
TEST_GAMES = 100
outputFilePath = "TestSave/testOutput_80percentWin_SGD_2Dense_Batch_32.txt"
import sys




def syncAgentsWeights(learningAgent, opponentAgent):
    opponentAgent.model.set_weights(learningAgent.model.get_weights())

def writeStdOutputToFile(filePath, text):
    print(text)
    original_std_out = sys.stdout
    with open(filePath, "a") as f:
        sys.stdout = f
        print(text)
        sys.stdout = original_std_out


def playTestGames(gamesNumber):
    envTest = ReversiEnv("random", "numpy3c", "lose", 8)
    agent = DDQNAgent(state_size, envTest, 0)
    agent.load("Save/learning_agent_model_weights_trained.h5")
    agent.epsilon = 0
    opponent = RandomAgent(state_size, action_size, envTest, 1)
    games_won = 0
    test_win_percentage = 0.
    stateTest = envTest.reset()
    test_episodes_counter = 0
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

        envTest.currently_playing_color = opponent.player_color

        envTest.possible_actions = ReversiEnv.get_possible_actions(envTest.state, envTest.currently_playing_color)

        if len(envTest.possible_actions) != 0:
            opponentAgent_action = opponent.get_action_to_make(stateTest)
            next_state_test, reward2, done, _ = envTest.step(opponentAgent_action)
            next_state_test = np.reshape(next_state_test, [1, state_size])

        else:
            envTest.pass_place_counter += 1

        if envTest.pass_place_counter > 1:
            done = True

        stateTest = next_state_test

        if done:
            test_episodes_counter += 1
            black_score_test = len(np.where(envTest.state[0, :, :] == 1)[0])
            white_score_test = len(np.where(envTest.state[1, :, :] == 1)[0])

            if black_score_test > white_score_test:
                games_won += 1
                test_win_percentage = games_won / test_episodes_counter * 100
                writeStdOutputToFile(outputFilePath, f"Test game {test_episodes_counter} result - B/W: {black_score_test}/{white_score_test} "
                                                     f"Test games win percentage: {test_win_percentage} "
                                                     f"Test games won: {games_won} "
                                                     f"Test games lost: {test_episodes_counter - games_won} ")
            else:
                test_win_percentage = games_won / test_episodes_counter * 100

                writeStdOutputToFile(outputFilePath, f"Test game {test_episodes_counter} result - B/W: {black_score_test}/{white_score_test} "
                                                     f"Test games win percentage: {test_win_percentage} "
                                                     f"Test games won: {games_won} "
                                                     f"Test games lost: {test_episodes_counter - games_won}")

            stateTest = envTest.reset()
        if test_episodes_counter == gamesNumber:
            break
    return test_win_percentage


#########################################################################################################################################################################
############### MAIN LOOP ###############################################################################################################################################
#########################################################################################################################################################################


if __name__ == '__main__':
    episodes_counter = 0
    env = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learningAgent = DDQNAgent(state_size, env, 0)
    opponentAgent = RandomAgent(state_size, action_size, env, 1)
    games_won = 0
    win_percentage_overall = 0.
    test_games_win_percentage = 0.
    best_test_games_win_percentage = 0.
    state = env.reset()

    writeStdOutputToFile(outputFilePath, f"Hyperparameters - Learning rate: {learningAgent.learning_rate}, "
                                         f"replay buffer size: {learningAgent.replay_buffer_size}, "
                                         f"gamma: {learningAgent.gamma}, \n"
                                         f"epsilon min: {learningAgent.epsilon_min},"
                                         f" epsilon decay: {learningAgent.epsilon_decay}, "
                                         f"batch size: {BATCH_SIZE}")

    while True:

        learningAgent_action = None
        opponentAgent_action = None

        state = np.reshape(state, [1, state_size])
        env.currently_playing_color = learningAgent.player_color

        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)
        if len(env.possible_actions) != 0:
            learningAgent_action = learningAgent.get_action_to_make(state)
            next_state, reward, done, _ = env.step(learningAgent_action)
            next_state = np.reshape(next_state, [1, state_size])

            if done != True:
                learningAgent.replay_buffer_save(state, learningAgent_action, reward, next_state, done)

            state = next_state

        else:
            env.pass_place_counter += 1


        env.currently_playing_color = opponentAgent.player_color

        env.possible_actions = ReversiEnv.get_possible_actions(env.state, 1)

        if len(env.possible_actions) != 0:
            opponentAgent_action = opponentAgent.get_action_to_make(state)
            next_state, reward, done, _ = env.step(opponentAgent_action)

            next_state = np.reshape(next_state, [1, state_size])

            state = next_state

        else:
            env.pass_place_counter += 1

        if env.pass_place_counter > 1:
            done = True



        if done:
            episodes_counter += 1

            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])

            if black_score > white_score:
                reward_done = 10.
            else:
                reward_done = -10.
            learningAgent.replay_buffer_save(state, learningAgent_action, reward_done, next_state, done)

            if len(learningAgent.memory) > BATCH_SIZE and episodes_counter > 50:
                learningAgent.replay(BATCH_SIZE)



            if black_score > white_score:
                games_won += 1
                win_percentage_overall = games_won / episodes_counter * 100




                writeStdOutputToFile(outputFilePath,
                                     f"Games won/total: {games_won}/{episodes_counter},"
                                     f" win %: {win_percentage_overall:.4}%,"
                                     f" last score - black/white:{black_score}/{white_score},"
                                     f" learning agent epsilon: {learningAgent.epsilon},")


            else:

                win_percentage_overall = games_won / episodes_counter * 100
                writeStdOutputToFile(outputFilePath,
                                     f"Games won/total: {games_won}/{episodes_counter},"
                                     f" win %: {win_percentage_overall:.4}%,"
                                     f" last score - black/white:{black_score}/{white_score},"
                                     f" learning agent epsilon: {learningAgent.epsilon},")

            learningAgent.model.save(f"TestSave/learning_agent_model_weights_trained.h5")


            if episodes_counter % 50 == 0:

                learningAgent.sync_target_model()

                writeStdOutputToFile(outputFilePath, "Evaluate model!")

                test_games_win_percentage = playTestGames(TEST_GAMES)
                if test_games_win_percentage >= best_test_games_win_percentage:
                    best_test_games_win_percentage = test_games_win_percentage
                    writeStdOutputToFile(outputFilePath, "Saving model weights!")
                    learningAgent.save(f"TestSave/model_weights_{test_games_win_percentage}.h5")
                    learningAgent.target_model.save(
                        f"TestSave/target_model_weights_final_{test_games_win_percentage}.h5")

                    if best_test_games_win_percentage > 80:
                        break

            state = env.reset()