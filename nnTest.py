from Reversi_combined import ReversiEnv
from Agents.NNAgent import DDQNAgent
import random
from Agents.MinMaxAgent import MinMaxAgent
from copy import deepcopy
import numpy as np
EPISODES = 100
BATCH_SIZE = 50
outputFilePath = "TestSave/output.txt"
import sys

def syncAgentsWeights(learningAgent, opponentAgent):
    opponentAgent.target_model.set_weights(learningAgent.target_model.get_weights())

def writeStdOutputToFile(filePath, text):
    original_std_out = sys.stdout
    with open(filePath, "a") as f:
        sys.stdout = f
        print(text)
        sys.stdout = original_std_out

if __name__ == '__main__':
    episodes_counter = 0
    env = ReversiEnv("random", "numpy3c", "lose", 8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learningAgent = DDQNAgent(state_size, action_size, env, 0)
    opponentAgent = DDQNAgent(state_size, action_size, env, 1)
    games_won = 0
    state = env.reset()
    #state = env.reset()
    env.render()
    while True:
        if episodes_counter == EPISODES:
            break
        state = np.reshape(state, [1, state_size])
        env.currently_playing_color = learningAgent.player_color
        #tmp_state = deepcopy(env.state)
        #state = env.state
        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)
        learningAgent_action = learningAgent.get_action_to_make(state)
        next_state, reward, done, _ = env.step(learningAgent_action)
        next_state = np.reshape(next_state, [1, state_size])
        #state = np.reshape(state, [1, state_size])
        learningAgent.replay_buffer_save(state, learningAgent_action, reward, next_state, done)
        state = next_state
        env.render()
        env.currently_playing_color = opponentAgent.player_color
        env.possible_actions = ReversiEnv.get_possible_actions(env.state, env.currently_playing_color)
        opponentAgent_action = opponentAgent.get_action_to_make(state)
        next_state, _, done, _ = env.step(opponentAgent_action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        env.render()
        if len(learningAgent.memory) > BATCH_SIZE:
            learningAgent.replay(BATCH_SIZE)
        if done:
            episodes_counter += 1

            #env.reset()
            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])
            if reward > 0:
                games_won += 1
                learningAgent.save("TestSave/model_weights.h5")
                learningAgent.target_model.save("TestSave/target_model_weights.h5")

                print(
                    "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}".format(
                                                                                    games_won, episodes_counter,
                                                                                    games_won / episodes_counter * 100,
                                                                                    black_score, white_score))
                writeStdOutputToFile(outputFilePath, "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}".format(
                    games_won, episodes_counter,
                    games_won / episodes_counter * 100,
                    black_score, white_score))
            else:

                print(
                    "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}".format(
                                                                                    games_won,
                                                                                    episodes_counter,
                                                                                    games_won / episodes_counter * 100,
                                                                                    black_score, white_score))
                writeStdOutputToFile(outputFilePath,
                                     "Games won/total: {}/{}, win %: {:.4}%, last score - black/white:{}/{}".format(
                                        games_won, episodes_counter,
                                         games_won / episodes_counter * 100,
                                         black_score, white_score))

            state = env.reset()
            if episodes_counter % 10 == 0:
                print("Syncing agents weights...")
                writeStdOutputToFile(outputFilePath, "Syncing agents weights...")
                syncAgentsWeights(learningAgent, opponentAgent)
                opponentAgent.epsilon = 0



        # env.render()
        # env.currently_playing_color = opponentAgent.player_color
        # tmp_state = deepcopy(env.state)
        # opponentAgent_action = opponentAgent.best_action(tmp_state)
        # env.step(opponentAgent_action)
        # env.render()










