from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
from collections import deque
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 1000
BATCH_SIZE = 40
outputFilePath = "TestSave/relu_activation_SGD_optimizer_score_as_reward.txt"
import sys

def syncAgentsWeights(learningAgent, opponentAgent):
    opponentAgent.model.set_weights(learningAgent.model.get_weights())

def writeStdOutputToFile(filePath, text):
    original_std_out = sys.stdout
    with open(filePath, "a") as f:
        sys.stdout = f
        print(text)
        sys.stdout = original_std_out

if __name__ == '__main__':
    episodes_counter = 0
    won_in_row = 0 #deque(maxlen=10)
    env = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learningAgent = DDQNAgent(state_size, action_size, env, 0)
    opponentAgent = DDQNAgent(state_size, action_size, env, 1)
    games_won = 0
    win_percentage_overall = 0.
    #last_ten_win_percentage = 0.
    best_won_in_row = 0.
    state = env.reset()
    #state = env.reset()
    print(
        f"Hyperparameters - Learning rate: {learningAgent.learning_rate}, replay buffer size: {learningAgent.replay_buffer_size}, gamma: {learningAgent.gamma}, \n"
        f"epsilon min: {learningAgent.epsilon_min}, epsilon decay: {learningAgent.epsilon_decay}, batch size: {BATCH_SIZE}")
    writeStdOutputToFile(outputFilePath, f"Hyperparameters - Learning rate: {learningAgent.learning_rate}, replay buffer size: {learningAgent.replay_buffer_size}, gamma: {learningAgent.gamma}, \n"
        f"epsilon min: {learningAgent.epsilon_min}, epsilon decay: {learningAgent.epsilon_decay}, batch size: {BATCH_SIZE}")
    #env.render()

    while True:

        reward1 = 0
        reward2 = 0
        state1 = None
        learningAgent_action = None
        if episodes_counter == EPISODES:
            break
        state = np.reshape(state, [1, state_size])
        env.currently_playing_color = learningAgent.player_color
        #tmp_state = deepcopy(env.state)
        #state = env.state
        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)
        if len(env.possible_actions) != 0:
            learningAgent_action = learningAgent.get_action_to_make(state)
            next_state, reward1, done, _ = env.step(learningAgent_action)
            next_state = np.reshape(next_state, [1, state_size])
            #state = np.reshape(state, [1, state_size])
            #learningAgent.replay_buffer_save(state, learningAgent_action, reward1, next_state, done)
            state1 = state
            state = next_state

        else:
            env.pass_place_counter += 1


        #env.render()
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

        #env.render()
        if len(learningAgent.memory) > BATCH_SIZE:
            learningAgent.replay(BATCH_SIZE)
        if done:
            learningAgent.sync_target_model()
            episodes_counter += 1

            #env.reset()
            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])

            if black_score>white_score:
                won_in_row += 1 #.append(1)
                games_won += 1
                win_percentage_overall = games_won / episodes_counter * 100
                #last_ten_win_percentage = won_in_row.count(1) * 10

                print(
                    "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}, learning agent epsilon: {}, games won in a row: {}".format(
                                                                                    games_won, episodes_counter,
                                                                                    games_won / episodes_counter * 100,
                                                                                    black_score, white_score, learningAgent.epsilon, won_in_row))
                writeStdOutputToFile(outputFilePath, "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}, learning agent epsilon: {}, games won in a row: {}".format(
                    games_won, episodes_counter,
                    win_percentage_overall,
                    black_score, white_score, learningAgent.epsilon, won_in_row))
                if won_in_row > best_won_in_row:
                    best_won_in_row = won_in_row
                    learningAgent.save("TestSave/model_weights.h5")
                    learningAgent.target_model.save("TestSave/target_model_weights.h5")
                    print("Syncing agents weights...")
                    writeStdOutputToFile(outputFilePath, "Syncing agents weights...")
                    syncAgentsWeights(learningAgent, opponentAgent)
                    #if best_won_in_row == 100:
                     #   best_won_in_row = 0
            else:
                won_in_row = 0
                #won_in_row.append(0)
                #last_ten_win_percentage = won_in_row.count(1) * 10
                print(
                    "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}, learning agent epsilon: {}, games won in a row: {}".format(
                                                                                    games_won,
                                                                                    episodes_counter,
                                                                                    games_won / episodes_counter * 100,
                                                                                    black_score, white_score, learningAgent.epsilon, won_in_row))
                writeStdOutputToFile(outputFilePath,
                                     "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}, learning agent epsilon: {}, games won in a row: {}".format(
                                        games_won, episodes_counter,
                                         win_percentage_overall,
                                         black_score, white_score, learningAgent.epsilon, won_in_row))
                # if last_ten_win_percentage >= best_win_percentage:
                #     best_win_percentage = last_ten_win_percentage
                #     print("Syncing agents weights...")
                #     writeStdOutputToFile(outputFilePath, "Syncing agents weights...")
                #     syncAgentsWeights(learningAgent, opponentAgent)
                #     opponentAgent.epsilon = 0



            state = env.reset()
            #if episodes_counter % 10 == 0:
                 # print("Syncing agents weights...")
                 # writeStdOutputToFile(outputFilePath, "Syncing agents weights...")
                 # syncAgentsWeights(learningAgent, opponentAgent)
                 # opponentAgent.epsilon = 0



        # env.render()
        # env.currently_playing_color = opponentAgent.player_color
        # tmp_state = deepcopy(env.state)
        # opponentAgent_action = opponentAgent.best_action(tmp_state)
        # env.step(opponentAgent_action)
        # env.render()










