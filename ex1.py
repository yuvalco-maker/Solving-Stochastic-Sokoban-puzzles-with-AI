# github co-pilot was used in writing this code i required help in pythons syntax as im not familiar with it
import search
import utils

id = ["211602297"]

BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
flag = 0

class PressurePlateProblem(search.Problem):
    def __init__(self, initial):
        self.map = [list(row) for row in initial]
        self.original_map = tuple(tuple(row) for row in self.map)
        initial = self.original_map
        self.distance_to_goal = self.compute_distance_to_goal()
        self.plate_locations = {k: [] for k in range(10)}
        for i in range(len(self.original_map)):
            for j in range(len(self.original_map[0])):
                val = self.original_map[i][j]
                if 20 <= val <= 29:
                    self.plate_locations[val - 20].append((i, j))
        super().__init__(initial)

    def compute_distance_to_goal(self):
        state = self.map
        n, m = len(state), len(state[0])
        distance = {}
        goal = None
        for i in range(n):
            for j in range(m):
                if state[i][j] == GOAL:
                    goal = (i, j)
                    break
            if goal:
                break
        if not goal:
            return {}
        queue = [goal]
        distance[goal] = 0
        queue_index = 0
        while queue_index < len(queue):
            x, y = queue[queue_index]
            queue_index += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and state[nx][ny] != WALL and (nx, ny) not in distance:
                    distance[(nx, ny)] = distance[(x, y)] + 1
                    queue.append((nx, ny))
        return distance

    def is_obstacle(self, state, x, y, key_val):
        if not (0 <= x < len(state) and 0 <= y < len(state[0])):
            return True
        cell = state[x][y]
        if cell == WALL or self.is_locked_door(cell):
            return True
        if self.is_block(cell) and cell != key_val:
            return True
        if 20 <= cell <= 29 and cell != key_val + 10:
            return True
        if 30 <= cell <= 39 and cell != key_val + 20:
            return True
        return False

    def is_stuck(self, state, x, y):
        key_val = state[x][y]
        return (
            (self.is_obstacle(state, x - 1, y, key_val) and self.is_obstacle(state, x, y - 1, key_val)) or
            (self.is_obstacle(state, x - 1, y, key_val) and self.is_obstacle(state, x, y + 1, key_val)) or
            (self.is_obstacle(state, x + 1, y, key_val) and self.is_obstacle(state, x, y - 1, key_val)) or
            (self.is_obstacle(state, x + 1, y, key_val) and self.is_obstacle(state, x, y + 1, key_val))
        )

    def is_block(self, val):
        return 10 <= val <= 19

    def is_plate(self, val):
        return 20 <= val <= 29

    def is_pressed_plate(self, val):
        return 30 <= val <= 39

    def is_locked_door(self, val):
        return 40 <= val <= 49

    def unlock(self, state):
        state_list = [list(row) for row in state]
        n, m = len(self.original_map), len(self.original_map[0])
        for k in range(10):
            pressed_type = 30 + k
            door_type = 40 + k
            plate_locations = self.plate_locations[k]
            if not plate_locations:
                continue
            all_pressed = all(state[i][j] == pressed_type for i, j in plate_locations)
            if all_pressed:
                for i in range(n):
                    for j in range(m):
                        if state_list[i][j] == door_type:
                            state_list[i][j] = FLOOR
        return tuple(tuple(row) for row in state_list)

    def can_push(self, state, bx, by, dx, dy):
        nx, ny = bx + dx, by + dy
        if not (0 <= nx < len(state) and 0 <= ny < len(state[0])):
            return False
        target = state[nx][ny]
        block_val = state[bx][by]
        flag = 1
        return (
            target in [FLOOR, BLANK, GOAL] or
            (self.is_plate(target) and target == block_val + 10)
        )

    def successor(self, state):
        directions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        successors = []
        ax = ay = None
        n, m = len(state), len(state[0])
        for i, row in enumerate(state):
            for j, val in enumerate(row):
                if val == AGENT or val == AGENT_ON_GOAL:
                    ax, ay = i, j
                    break
            if ax is not None:
                break

        for action, (dx, dy) in directions.items():
            nx, ny = ax + dx, ay + dy
            if not (0 <= nx < n and 0 <= ny < m):
                continue
            target = state[nx][ny]

            if self.is_block(target) and self.can_push(state, nx, ny, dx, dy):
                nnx, nny = nx + dx, ny + dy
                if not (0 <= nnx < n and 0 <= nny < m):
                    continue
                new_state = [list(row) for row in state]
                if self.is_plate(state[nnx][nny]):
                    new_state[nnx][nny] = state[nx][ny] + 20
                else:
                    new_state[nnx][nny] = state[nx][ny]
                new_state[nx][ny] = AGENT
                new_state[ax][ay] = GOAL if state[ax][ay] == AGENT_ON_GOAL else FLOOR
                new_state = self.unlock(tuple(tuple(row) for row in new_state))
                if new_state != state:
                    successors.append((action, new_state))

            elif target in [FLOOR, BLANK] or target == GOAL:
                new_state = [list(row) for row in state]
                new_state[ax][ay] = GOAL if state[ax][ay] == AGENT_ON_GOAL else FLOOR
                new_state[nx][ny] = AGENT_ON_GOAL if target == GOAL else AGENT
                new_state = self.unlock(tuple(tuple(row) for row in new_state))
                if new_state != state:
                    successors.append((action, new_state))

        return successors

    def goal_test(self, state):
        return all(val != GOAL for row in state for val in row)

    @staticmethod
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def h(self, node):
        state = node.state
        xa = ya = None
        n, m = len(state), len(state[0])
        for i in range(n):
            for j in range(m):
                if state[i][j] == AGENT or state[i][j] == AGENT_ON_GOAL:
                    xa, ya = i, j
                    break
            if xa is not None:
                break

        if xa is None or (xa, ya) not in self.distance_to_goal:
            return 10000

        dist = self.distance_to_goal[(xa, ya)]

        block_to_plate_dist = 0
        for k in range(10):
            block_type = 10 + k
            blocks = [(i, j) for i in range(n)
                      for j in range(m)
                      if state[i][j] == block_type]
            plates = self.plate_locations[k]
            plates_not_pressed = [p for p in plates if state[p[0]][p[1]] != 30 + k]
            for block in blocks:
                if plates_not_pressed:
                    min_dist = min(self.manhattan(block, p) for p in plates_not_pressed)
                    block_to_plate_dist += min_dist

        return dist + 0.5 * block_to_plate_dist


def create_pressure_plate_problem(game):
    print("<<create_pressure_plate_problem")
    return PressurePlateProblem(game)

