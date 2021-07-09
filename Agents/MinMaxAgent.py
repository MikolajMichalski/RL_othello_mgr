from Reversi_combined import ReversiEnv
from copy import deepcopy
import random
class MinMaxAgent:

    def __init__(self, env, max_depth, player_color):
        self.env = env
        self.max_depth = max_depth
        self.player_color = player_color

    def calculate_min_max_action(self, game_state, depth, maximizing_player):
        tmp_state = deepcopy(game_state)

        if maximizing_player:
            current_player_color = 1
        else:
            current_player_color = 0
        possible_actions_tmp = ReversiEnv.get_possible_actions(tmp_state, current_player_color)

        if depth == self.max_depth or (len(possible_actions_tmp)==1 and any(possible_actions_tmp) > 64):
            tmp_obs = ReversiEnv.getCurrentObservations(tmp_state)
            score = sum(1 for i in ReversiEnv.getCurrentObservations(tmp_state) if i==-1)
            return score

        if maximizing_player:
            best_score = -9999
            possible_actions_tmp = ReversiEnv.get_possible_actions(tmp_state, current_player_color)
            for action in possible_actions_tmp:
                tmp_state1 = deepcopy(tmp_state)
                if action != 65:
                    tmp_state = ReversiEnv.make_place(tmp_state, action, 1)
                score = self.calculate_min_max_action(tmp_state, depth + 1, False)
                tmp_state = tmp_state1
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = 9999
            possible_actions_tmp = ReversiEnv.get_possible_actions(tmp_state, current_player_color)
            for action in possible_actions_tmp:
                tmp_state1 = deepcopy(tmp_state)
                if action != 65:
                    tmp_state = ReversiEnv.make_place(tmp_state, action, 0)
                score = self.calculate_min_max_action(tmp_state, depth+1, True)
                tmp_state = tmp_state1
                best_score = min(score, best_score)

            return best_score



    def best_action(self, game_state):

        best_score = -9999
        tmp_state = game_state
        possible_actions_tmp = deepcopy(ReversiEnv.get_possible_actions(tmp_state, self.player_color))
        action_score_dict = dict()
        for action in possible_actions_tmp:
            tmp_state1 = deepcopy(tmp_state)
            if action != 65:
                tmp_state = ReversiEnv.make_place(tmp_state, action, self.player_color)

            score = self.calculate_min_max_action(tmp_state, 0, False)
            tmp_state = tmp_state1
            action_score_dict.update({action : score})
            if score > best_score:
                best_score = score
                b_action = action
            action_score_dict_sorted = sorted(action_score_dict.items(), key=lambda kv: kv[1])
            k_best_actions = action_score_dict_sorted[-2::]
            selected_action = k_best_actions[random.randint(0, len(k_best_actions)-1)][0]
            b_action = selected_action
        return b_action


