import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from enum import IntEnum


# using IntEnum will make the code a lot more descriptive and readable
class CellType(IntEnum):
    EMPTY = 0
    FOOD = 1
    WATER = 2
    SHELTER = 3
    DANGER = 4
    OBSTACLE = 5


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4
    DRINK = 5


class DontStarveSingleAgentEnv(gym.Env):
    def __init__(
        self, grid_size: int = 10, max_days: int = 10, render_mode: str = "ansi"
    ):  # too lazy to look into human and rgb for now
        """_summary_

        Args:
            grid_size (int, optional): length of the axis on the 2D Grid. Defaults to 10.
            max_days (int, optional): The max duration that the game lasts for. Defaults to 10.
            render_mode (str, optional): rendering settings. Defaults to "ansi".
        """
        super().__init__()  # inherits core attributes, e.g., reward_range, metadata, np_random
        self.grid_size = grid_size
        # define observation space even though it's not used in this example
        # the framework will use the observation space to validate the observation
        # returned by the environment
        self.observation_space = spaces.Box(low=0, high=6, shape=(grid_size, grid_size))
        self.max_days = max_days
        self.hp = 100
        self.satiety = 100
        self.hydration = 100
        self.days = 1
        self.inventory = {"food": 10, "water": 10}
        self.depletion_rate = {"food": 1, "water": 1}
        self.action_space = spaces.Discrete(len(Action))
        self.render_mode = render_mode
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    def step(self, action):

        # initialize the cumulative reward for this step
        reward = 0

        current_day = self.days

        # ----------------- Action logic -----------------

        # Moving up in y-axis
        if action == Action.UP and self.player_position[0] > 0:
            self.player_position = (
                self.player_position[0] - 1,
                self.player_position[1],
            )
        # Moving down in y-axis
        elif action == Action.DOWN and self.player_position[0] < self.grid_size - 1:
            self.player_position = (
                self.player_position[0] + 1,
                self.player_position[1],
            )
        # Moving left in x-axis
        elif action == Action.LEFT and self.player_position[1] > 0:
            self.player_position = (
                self.player_position[0],
                self.player_position[1] - 1,
            )
        # Moving right in x-axis
        elif action == Action.RIGHT and self.player_position[1] < self.grid_size - 1:
            self.player_position = (
                self.player_position[0],
                self.player_position[1] + 1,
            )

        # Choose to eat or drink in the current cell
        elif action == Action.EAT:
            if self.inventory["food"] > 0:
                self.inventory["food"] -= 1
                self.satiety = min(100, self.satiety + 10)
                self.hp += min(100, self.hp + 10)
            else:
                self.hp -= 5  # Added damage from hunger
                reward -= 0.5  # Penalty for not having food when trying to eat
        elif action == Action.DRINK:
            if self.inventory["water"] > 0:
                self.inventory["water"] -= 1
                self.hydration = min(100, self.hydration + 10)
            else:
                self.hydration -= 5  # Added damage from thirst
                reward -= 0.5  # Penalty for not having water when trying to drink

        y, x = self.player_position

        # info dictionary for storing and managing state information
        info = {
            # ----------- Player state ----------------
            "player_position": self.player_position,
            "hp": self.hp,
            "satiety": self.satiety,
            "hydration": self.hydration,
            "inventory": self.inventory.copy(),
            # ----------- Game state ----------------
            "day": self.days,
            "action_taken": Action(action).name,  # Convert int to enum name
            # ----------- Game state ----------------
            "moved": action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
            "consumed_resource": action in [Action.EAT, Action.DRINK],
            "current_cell_type": CellType(self.grid[y, x]).name,
            "cells_visited": int(np.sum(self.visited)),
            "percent_explored": float(np.sum(self.visited))
            / (self.grid_size * self.grid_size),
        }

        # Additional flags
        if self.inventory["food"] == 0:
            info["out_of_food"] = True

        if self.inventory["water"] == 0:
            info["out_of_water"] = True

        if self.hp < 20:
            info["critical_hp"] = True

        # Exploration reward

        if not self.visited[y, x]:
            self.visited[y, x] = True
            reward += 0.5

        # Collecting resources reward
        if self.grid[y, x] == CellType.FOOD or self.grid[y, x] == CellType.WATER:
            reward += 0.5

        # Penalties for dangerous cells
        if self.grid[y, x] == CellType.DANGER:
            reward -= 3.0
            self.hp -= 5  # Added damage from danger

        if self.grid[y, x] == CellType.OBSTACLE:
            reward -= 1.0

        if self.grid[y, x] == CellType.SHELTER:
            reward += 5.0
            self.hp = min(100, self.hp + 10)
            self.satiety = min(100, self.satiety + 10)
            self.hydration = min(100, self.hydration + 10)

        # Penalties for poor health
        if self.hp < 20 or self.hydration < 20 or self.satiety < 20:
            reward -= 0.5  # Penalty for getting close to death

        # Resource collection logic
        if self.grid[y, x] == CellType.FOOD:
            self.inventory["food"] += 1
            self.grid[y, x] = CellType.EMPTY
        elif self.grid[y, x] == CellType.WATER:
            self.inventory["water"] += 1
            self.grid[y, x] = CellType.EMPTY

        # Resource depletion
        # you will become hungry and thirsty over time
        self.satiety -= self.depletion_rate["food"]
        self.hydration -= self.depletion_rate["water"]
        self.hp -= self.depletion_rate["food"] + self.depletion_rate["water"]

        # Survived another day
        self.days += 1

        # Survival reward
        if self.days > current_day and not (
            self.hp <= 0 or self.satiety <= 0 or self.hydration <= 0
        ):
            reward += 2.0  # Bigger reward for surviving a full day
            info["survived_day"] = current_day

        # Terminal state checks
        terminated = False
        truncated = False

        if self.hp <= 0 or self.satiety <= 0 or self.hydration <= 0:
            terminated = True
            reward -= 10.0  # Extra penalty for dying

        if self.days >= self.max_days:
            truncated = True
            reward += 5.0  # Bonus for reaching max days

        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # use the base class reset method
        self.hp = 100
        self.satiety = 100
        self.hydration = 100
        self.days = 1
        self.inventory = {"food": 10, "water": 10}
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        # Place the player in the center of the grid
        self.player_position = (self.grid_size // 2, self.grid_size // 2)

        # Use a set to track the cells with resources so we don't place them on top of each other
        cells_with_resources = set()

        for _ in range(self.grid_size):
            # Place food
            while True:
                x, y = random.randint(0, self.grid_size - 1), random.randint(
                    0, self.grid_size - 1
                )
                if (y, x) not in cells_with_resources and (
                    y,
                    x,
                ) != self.player_position:
                    self.grid[y, x] = CellType.FOOD
                    cells_with_resources.add((y, x))
                    break

            # Place water
            while True:
                x, y = random.randint(0, self.grid_size - 1), random.randint(
                    0, self.grid_size - 1
                )
                if (y, x) not in cells_with_resources and (
                    y,
                    x,
                ) != self.player_position:
                    self.grid[y, x] = CellType.WATER
                    cells_with_resources.add((y, x))
                    break

        # Place obstacles (fewer than food/water)
        for _ in range(self.grid_size // 3):
            while True:
                x, y = random.randint(0, self.grid_size - 1), random.randint(
                    0, self.grid_size - 1
                )
                if (y, x) not in cells_with_resources and (
                    y,
                    x,
                ) != self.player_position:
                    self.grid[y, x] = CellType.OBSTACLE
                    cells_with_resources.add((y, x))
                    break

        # Place danger zones (sparse)
        for _ in range(self.grid_size // 4):
            while True:
                x, y = random.randint(0, self.grid_size - 1), random.randint(
                    0, self.grid_size - 1
                )
                if (y, x) not in cells_with_resources and (
                    y,
                    x,
                ) != self.player_position:
                    self.grid[y, x] = CellType.DANGER
                    cells_with_resources.add((y, x))
                    break
        # Place shelter (rare)
        for _ in range(max(1, self.grid_size // 8)):
            while True:
                x, y = random.randint(0, self.grid_size - 1), random.randint(
                    0, self.grid_size - 1
                )
                if (y, x) not in cells_with_resources and (
                    y,
                    x,
                ) != self.player_position:
                    self.grid[y, x] = CellType.SHELTER
                    cells_with_resources.add((y, x))
                    break

        return self.observation, {}

    def render(self):
        if self.render_mode is None:
            return None
        # Text-based rendering
        if self.render_mode == "ansi":
            output = ""
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (y, x) == self.player_position:
                        output += "P "
                    else:
                        cell = self.grid[y, x]
                        if cell == CellType.EMPTY:
                            output += ". "
                        elif cell == CellType.FOOD:
                            output += "F "
                        elif cell == CellType.WATER:
                            output += "W "
                        elif cell == CellType.SHELTER:
                            output += "S "
                        elif cell == CellType.DANGER:
                            output += "D "
                        elif cell == CellType.OBSTACLE:
                            output += "O "
                output += "\n"
            return output

    @property
    def observation(self):
        obs = self.grid.copy()
        player_y, player_x = self.player_position
        player_grid = obs.copy()
        player_grid[player_y, player_x] = 6  # Value for player
        return player_grid
