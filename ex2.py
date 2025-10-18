# ai tools were assisted in solving this problem
# me and the student tom sasson shared ideas and approches to the problem
import pressure_plate
import numpy as np
import ex1
import search
import copy
import random
from collections import deque
from ex1 import create_pressure_plate_problem
from ex1 import PressurePlateProblem
from search import astar_search

id = ["211602297"]

BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
REPEAT = 40
DISCOUNT = 0.9
STUCK_PENALTY = -75
OPT_BONUS = 150
DEAD_END_PENALTY = -75
MAX_DEPTH = 4
LIMIT = 10

ACTIONS = ['U', 'L', 'R', 'D']
SWAP = {'U': 'D', 'D': 'U'}
DET_PROB = {
    'U': [1.0, 0.0, 0.0, 0.0],
    'D': [0.0, 0.0, 0.0, 1.0],
    'L': [0.0, 1.0, 0.0, 0.0],
    'R': [0.0, 0.0, 1.0, 0.0]
}

class Controller:
    def __init__(self, game: pressure_plate.Game):
        self.original_game = game
        model = game.get_model()
        self.probs = model['chosen_action_prob']
        self.s_cost = model['step_punishment']
        self.finished_reward = model['finished_reward']
        self.opening_door_reward = model['opening_door_reward']

        self.optimal_path_nodes = []
        self.optimal_games = []
        self.optimal_actions = []
        self.all_children = []
        self.optimal_plates = []

        self.opt_path_policy = {}
        self.opt_diviation_policy = {}
        self.opt_values_path = {}
        self.opt_div_vals = {}

        self.current_step = 0
        self.resets = 0
        
        # Add caches
        self._hash_cache = {}
        self._heuristic_cache = {}
        self._obstacle_cache = {}
        self._stuck_cache = {}
        self._deadend_cache = {}
        
        # Pre-create a template game for faster copying
        self._template_game = copy.deepcopy(self.original_game)
        self._template_game._chosen_action_prob = DET_PROB

        self.problem = ex1.PressurePlateProblem(self.original_game.get_current_state()[0])
        self.optimal_actions = self.astar_actions(self.problem.map)
        self.optimal_games = self.build_astar_path_nodes()
        self.optimal_path_nodes = self.extract_astar_node_states()
        self.all_children = self.build_children_nodes()
        self.opt_values_path, self.opt_path_policy = self.get_optimal_paths_policy_and_values()
        self.opt_div_vals, self.opt_diviation_policy = self.find_opt_policy_and_values_for_children()

    def reset(self,game: pressure_plate.Game):
        self.resets += 1
        self.original_game = game
        self.optimal_path_nodes = []
        self.optimal_games = []
        self.optimal_actions = []
        self.all_children = []
        self.optimal_plates = []
        self.opt_path_policy = {}
        self.opt_diviation_policy = {}            
        self.opt_values_path = {}
        self.opt_div_vals = {}
        self.current_step = 0
        
        # Clear caches
        self._hash_cache.clear()
        self._heuristic_cache.clear()
        self._obstacle_cache.clear()
        self._stuck_cache.clear()
        self._deadend_cache.clear()
        
        # Update template
        self._template_game = copy.deepcopy(self.original_game)
        self._template_game._chosen_action_prob = DET_PROB
        
        self.problem = ex1.PressurePlateProblem(self.original_game.get_current_state()[0])
        self.optimal_actions = self.astar_actions(self.problem.map)
        self.optimal_games = self.build_astar_path_nodes()
        self.optimal_path_nodes = self.extract_astar_node_states()
        self.all_children = self.build_children_nodes()
        self.opt_values_path, self.opt_path_policy = self.get_optimal_paths_policy_and_values()
        self.opt_div_vals, self.opt_diviation_policy = self.find_opt_policy_and_values_for_children()

    def _get_hash(self, state_map):
        """Get hash of state map"""
        return tuple(state_map.flatten())

    def astar_actions(self, map):
        game = ex1.PressurePlateProblem(map)
        game._chosen_action_prob = DET_PROB
        result = astar_search(game)
        if result is None:
            return []
        goal_node, _ = result
        actions = []
        node = goal_node
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        swapped_results = [SWAP.get(a, a) for a in actions]
        return swapped_results

    def build_astar_path_nodes(self):
        if self.optimal_actions == []:
            return []
        game_copy = copy.deepcopy(self.original_game)
        game_copy._chosen_action_prob = DET_PROB
        opt_path = [copy.deepcopy(game_copy)]
        for action in self.optimal_actions:
            game_copy.submit_next_action(action)
            opt_path.append(copy.deepcopy(game_copy))
        return opt_path

    def extract_astar_node_states(self):
        return [game.get_current_state() for game in self.optimal_games]

    def get_next_position(self, x, y, action):
        if action == 'U':
            return x - 1, y
        elif action == 'D':
            return x + 1, y
        elif action == 'L':
            return x, y - 1
        elif action == 'R':
            return x, y + 1
        else:
            return x, y

    def dead_end(self, state):
        # Cache dead end checks
        cache_key = (self._get_hash(state[0]), state[1])
        if cache_key in self._deadend_cache:
            return self._deadend_cache[cache_key]
            
        map = state[0]
        x, y = state[1]
        obstacles = 0
        for action in ACTIONS:
            nx, ny = self.get_next_position(x, y, action)
            if self.is_obstacle(map, nx, ny, map[x][y]):
                obstacles += 1
        
        result = obstacles == 3
        self._deadend_cache[cache_key] = result
        return result

    def is_obstacle(self, state, x, y, key_val):
        # Cache obstacle checks
        cache_key = (x, y, key_val, self._get_hash(state))
        if cache_key in self._obstacle_cache:
            return self._obstacle_cache[cache_key]
            
        if not (0 <= x < len(state) and 0 <= y < len(state[0])):
            self._obstacle_cache[cache_key] = True
            return True
        cell = state[x][y]
        if cell == WALL or self.is_locked_door(cell):
            self._obstacle_cache[cache_key] = True
            return True
        if self.is_block(cell) and cell != key_val:
            self._obstacle_cache[cache_key] = True
            return True
        if 20 <= cell <= 29 and cell != key_val + 10:
            self._obstacle_cache[cache_key] = True
            return True
        if 30 <= cell <= 39 and cell != key_val + 20:
            self._obstacle_cache[cache_key] = True
            return True
        
        self._obstacle_cache[cache_key] = False
        return False

    def is_stuck(self, state, x, y):
        # Cache stuck checks
        cache_key = (self._get_hash(state), x, y)
        if cache_key in self._stuck_cache:
            return self._stuck_cache[cache_key]
            
        key_val = state[x][y]
        result = (
            (self.is_obstacle(state, x - 1, y, key_val) and self.is_obstacle(state, x, y - 1, key_val)) or
            (self.is_obstacle(state, x - 1, y, key_val) and self.is_obstacle(state, x, y + 1, key_val)) or
            (self.is_obstacle(state, x + 1, y, key_val) and self.is_obstacle(state, x, y - 1, key_val)) or
            (self.is_obstacle(state, x + 1, y, key_val) and self.is_obstacle(state, x, y + 1, key_val))
        )
        
        self._stuck_cache[cache_key] = result
        return result

    def is_block(self, val):
        return 10 <= val <= 19

    def is_plate(self, val):
        return 20 <= val <= 29

    def is_pressed_plate(self, val):
        return 30 <= val <= 39

    def is_locked_door(self, val):
        return 40 <= val <= 49

    def get_huristic_reward(self, state):
        # Cache heuristic rewards
        cache_key = (self._get_hash(state[0]), state[1])
        if cache_key in self._heuristic_cache:
            return self._heuristic_cache[cache_key]
            
        reward = 0
        map = state[0]
        x, y = state[1]
        reward += self.is_stuck(map, x, y) * STUCK_PENALTY
        hashed_key = self._get_hash(map)
        if hashed_key in self.opt_path_policy:
            reward += OPT_BONUS
        reward += self.dead_end(state) * DEAD_END_PENALTY
        
        self._heuristic_cache[cache_key] = reward
        return reward

    def build_children_nodes(self, max_depth=MAX_DEPTH):
        if self.optimal_games == []:
            return []
        children = []
        for game in self.optimal_games:
            seen_games = set()
            queue = deque([(copy.deepcopy(game), 0)])
            while queue:
                game, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                state = game.get_current_state()
                hashed_key = self._get_hash(state[0])
                if hashed_key in seen_games:
                    continue
                children.append(state)
                seen_games.add(hashed_key)
                for action in ACTIONS:
                    next_game = copy.deepcopy(game)
                    next_game.submit_next_action(action)
                    queue.append((next_game, depth + 1))
        return children

    def get_optimal_paths_policy_and_values(self):
        if self.optimal_path_nodes == []:
            return {}, {}
        optimal_values = {}
        optimal_policy = {}
        for i, state in enumerate(self.optimal_path_nodes[:-1]):
            hashed_map = self._get_hash(state[0])
            optimal_values[hashed_map] = i * 20 + 50
            optimal_policy[hashed_map] = self.optimal_actions[i]
        last_map = self.optimal_path_nodes[-1]
        hashed_map = self._get_hash(last_map[0])
        optimal_values[hashed_map] = len(self.optimal_path_nodes)
        optimal_policy[hashed_map] = 'D'
        return optimal_values, optimal_policy

    def find_opt_policy_and_values_for_children(self):
        if self.all_children == []:
            return {}, {}
        val = {}
        policy = {}
        for state in self.all_children:
            hashed_key = self._get_hash(state[0])
            val[hashed_key] = 0
            policy[hashed_key] = 'U'
        for _ in range(REPEAT):
            new_val = {}
            for state in self.all_children:
                value, action = self.value_iterration_step_for_state(state, val)
                hashed_key = self._get_hash(state[0])
                new_val[hashed_key] = value
                policy[hashed_key] = action
            val = new_val
        return val, policy

    def value_iterration_step_for_state(self, game_state, val):
        best_val = -np.inf
        best_action = 'L'
        
        # Pre-extract values
        map_state, agent_pos, steps, is_done, successful = game_state
        
        for action in ACTIONS:
            expectation = 0.0
            prob_vector = self.probs[action]
            
            for i, actual_action in enumerate(ACTIONS):
                p = prob_vector[i]
                if p == 0:  # Skip zero probabilities
                    continue
                    
                # Use shallow copy and only copy map
                sim_move = copy.copy(self._template_game)
                sim_move._map = map_state.copy()
                sim_move._agent_pos = agent_pos
                sim_move._steps = steps
                sim_move._done = is_done
                sim_move._successful = successful
                sim_move.submit_next_action(actual_action)
                
                state = sim_move.get_current_state()
                hashed_key = self._get_hash(state[0])
                value = val.get(hashed_key, -10)
                reward = sim_move.get_current_reward() + self.get_huristic_reward(state)
                expectation += p * (reward + DISCOUNT * value)
                
            if expectation > best_val:
                best_val = expectation
                best_action = action
        return best_val, best_action

    def choose_next_action(self, state):
        if self.optimal_actions == []:
            return random.choice(ACTIONS)
        hashed_key = self._get_hash(state[0])
        if hashed_key in self.opt_path_policy:
            return self.opt_path_policy[hashed_key]
        if hashed_key in self.opt_diviation_policy:
            return self.opt_diviation_policy[hashed_key]
        else:
            if self.resets > LIMIT:
                return random.choice(ACTIONS)
            else:
                game_copy = copy.deepcopy(self.original_game)
                game_copy._map = state[0]
                game_copy._agent_pos = state[1]
                game_copy._steps = state[2]
                game_copy._done = state[3]
                game_copy._successful = state[4]
                self.reset(game_copy)
            return self.choose_next_action(game_copy.get_current_state())