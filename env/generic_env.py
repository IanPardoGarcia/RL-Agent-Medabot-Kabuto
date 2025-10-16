import os
from pathlib import Path
import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict


ACTIONS = ["a", "b", "left", "right", "up", "down"]
TARGET_POSITIONS = [
    (16, 8, 80, 1),
    (17, 9, 80, 2),
    (16, 10, 80, 0),
]


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_STATE = DATA_DIR / "zero_state.state"


class GenericPyBoyEnv(Env):
    def __init__(self, pyboy, debug=False, render_mode=False, max_gameplay_time=1_080_000, state_path: Path = DEFAULT_STATE):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug
        self.render_mode = render_mode
        self.max_gameplay_time = max_gameplay_time
        self.current_gameplay_time = 0
        self.state_path = Path(state_path)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = Dict({
            "info": Box(0, 255, (4,), dtype=np.float32)
        })

        self.visited_maps = set()
        self.visited_positions = set()
        self.last_pos = None

        if not self.debug:
            # lower speed when rendering to keep UI responsive
            self.pyboy.set_emulation_speed(60 if not self.render_mode else 3)

        self.load_state()

    def load_state(self):
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}. Place zero_state.state inside data/.")
        with open(self.state_path, "rb") as f:
            # PyBoy.load_state accepts file-like objects
            self.pyboy.load_state(f)

    def get_observation(self):
        pos_x = int(self.pyboy.memory[0xC0D4])
        pos_y = int(self.pyboy.memory[0xC0D5])
        map_id = int(self.pyboy.memory[0xC92D])
        orientation = int(self.pyboy.memory[0xC0D8])
        return {"info": np.array([pos_x, pos_y, map_id, orientation], dtype=np.float32)}

    def step(self, action):
        # Accept both integer action or already mapped action strings
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0])
        elif isinstance(action, np.ndarray):
            action = int(action)

        self.pyboy.button(ACTIONS[action])
        for _ in range(60):
            self.pyboy.tick()
            self.current_gameplay_time += 1

        pos_x = int(self.pyboy.memory[0xC0D4])
        pos_y = int(self.pyboy.memory[0xC0D5])
        map_id = int(self.pyboy.memory[0xC92D])
        orientation = int(self.pyboy.memory[0xC0D8])
        full_pos = (pos_x, pos_y, map_id, orientation)

        reward = -0.001
        if map_id not in self.visited_maps:
            reward += 1.0
        if full_pos in TARGET_POSITIONS and action == 0:
            reward += 10.0

        self.visited_maps.add(map_id)
        self.visited_positions.add(full_pos)
        self.last_pos = full_pos

        terminated = False
        truncated = False
        if self.current_gameplay_time >= self.max_gameplay_time:
            truncated = True
        if full_pos in TARGET_POSITIONS and action == 0:
            terminated = True

        return self.get_observation(), float(reward), terminated, truncated, {}

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.reset_seed = seed
        self.load_state()
        self.current_gameplay_time = 0
        self.visited_maps.clear()
        self.visited_positions.clear()
        return self.get_observation(), {}

    def close(self):
        try:
            self.pyboy.stop()
        except Exception:
            pass
