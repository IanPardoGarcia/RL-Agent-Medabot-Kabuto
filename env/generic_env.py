import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict

ACTIONS = ['a', 'b', 'left', 'right', 'up', 'down']
TARGET_POSITIONS = [
    (16, 8, 80, 1),
    (17, 9, 80, 2),
    (16, 10, 80, 0)
]

class GenericPyBoyEnv(Env):
    def __init__(self, pyboy, debug=False, render_mode=False, max_gameplay_time=1_080_000):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug
        self.render_mode = render_mode
        self.max_gameplay_time = max_gameplay_time
        self.current_gameplay_time = 0

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = Dict({
            "info": Box(0, 255, (4,), dtype=np.float32)
        })

        self.visited_maps = set()
        self.visited_positions = set()
        self.last_pos = None

        if not self.debug:
            self.pyboy.set_emulation_speed(60 if not self.render_mode else 3)

        self.load_state()

    def load_state(self):
        with open("zero_state.state", "rb") as f:
            self.pyboy.load_state(f)

    def get_observation(self):
        pos_x = self.pyboy.memory[0xC0D4]
        pos_y = self.pyboy.memory[0xC0D5]
        map_id = self.pyboy.memory[0xC92D]
        orientation = self.pyboy.memory[0xC0D8]
        return {"info": np.array([pos_x, pos_y, map_id, orientation], dtype=np.float32)}

    def step(self, action):
        self.pyboy.button(ACTIONS[action])
        for _ in range(60):
            self.pyboy.tick()
            self.current_gameplay_time += 1

        pos_x = self.pyboy.memory[0xC0D4]
        pos_y = self.pyboy.memory[0xC0D5]
        map_id = self.pyboy.memory[0xC92D]
        orientation = self.pyboy.memory[0xC0D8]
        full_pos = (pos_x, pos_y, map_id, orientation)

        reward = -0.001
        if map_id not in self.visited_maps:
            reward += 1
        if full_pos in TARGET_POSITIONS and action == 0:
            reward += 10

        self.visited_maps.add(map_id)
        self.visited_positions.add(full_pos)
        self.last_pos = full_pos

        done = self.current_gameplay_time >= self.max_gameplay_time or (
            full_pos in TARGET_POSITIONS and action == 0
        )
        return self.get_observation(), reward, done, False, {}

    def reset(self, seed=None, **kwargs):
        self.load_state()
        self.current_gameplay_time = 0
        self.visited_maps.clear()
        self.visited_positions.clear()
        return self.get_observation(), {}

    def close(self):
        self.pyboy.stop()
