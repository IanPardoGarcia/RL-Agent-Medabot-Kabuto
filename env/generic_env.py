from pathlib import Path
import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict


# Action mapping used by PyBoy; these strings correspond to PyBoy.button() names
ACTIONS = ["a", "b", "left", "right", "up", "down"]

# Target positions are tuples of (x, y, map_id, orientation).
# These are used by the reward / termination logic to detect goal achievement.
TARGET_POSITIONS = [
    (16, 8, 80, 1),
    (17, 9, 80, 2),
    (16, 10, 80, 0),
]

# Data directory and default state file path
# The zero_state.state file should be placed inside data/ before running.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_STATE = DATA_DIR / "zero_state.state"

# Generic PyBoy environment compatible with Gymnasium interface.
class GenericPyBoyEnv(Env):
    def __init__(
        self,
        pyboy,
        debug: bool = False,
        render_mode: bool = False,
        max_gameplay_time: int = 1_080_000,
        state_path: Path = DEFAULT_STATE,
    ):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug
        self.render_mode = render_mode
        self.max_gameplay_time = max_gameplay_time
        self.current_gameplay_time = 0
        self.state_path = Path(state_path)

        # Discrete action space: index maps into ACTIONS above
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation is a small numeric vector: [pos_x, pos_y, map_id, orientation]
        # Values are in byte-range (0-255) so Box(0,255,(4,)) is used.
        self.observation_space = Dict({
            "info": Box(0, 255, (4,), dtype=np.float32)
        })

        self.visited_maps = set()
        self.visited_positions = set()
        self.last_pos = None

        if not self.debug:
            # Lower emulation speed for faster, headless runs.
            # When rendering (render_mode=True) we slow down the emulation to make
            # UI updates visible (3 fps here).
            self.pyboy.set_emulation_speed(60 if not self.render_mode else 3)

        self.load_state()

    def load_state(self):
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}. Place zero_state.state inside data/.")
        with open(self.state_path, "rb") as f:
            # PyBoy.load_state accepts file-like objects
            self.pyboy.load_state(f)

    def get_observation(self):
        # Memory offsets are Game Boy addresses observed empirically from the ROM.
        # These are "magic" values specific to this game; document and keep them together
        # to make future maintenance easier.
        pos_x = int(self.pyboy.memory[0xC0D4])
        pos_y = int(self.pyboy.memory[0xC0D5])
        map_id = int(self.pyboy.memory[0xC92D])
        orientation = int(self.pyboy.memory[0xC0D8])
        # Return the observation as a dict to match gymnasium.Dict observation space
        return {"info": np.array([pos_x, pos_y, map_id, orientation], dtype=np.float32)}

    def step(self, action):
        # Accept vector/array actions commonly returned by vectorized envs and
        # convert them to an integer index into ACTIONS
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(action[0])
        elif isinstance(action, np.ndarray):
            action = int(action)

        # Send button press to emulator and advance several ticks to let the game state update.
        # The "60" here is empirical (number of internal ticks to step); making this a
        # class parameter may help tune speed/response.
        self.pyboy.button(ACTIONS[action])
        for _ in range(60):
            self.pyboy.tick()
            self.current_gameplay_time += 1

        pos_x = int(self.pyboy.memory[0xC0D4])
        pos_y = int(self.pyboy.memory[0xC0D5])
        map_id = int(self.pyboy.memory[0xC92D])
        orientation = int(self.pyboy.memory[0xC0D8])
        full_pos = (pos_x, pos_y, map_id, orientation)

        # Reward design:
        # - small negative step penalty to encourage short solutions
        # - +1.0 for visiting a new map (map discovery)
        # - +10.0 for reaching a target position (and using action 0 — which corresponds to 'a')
        reward = -0.001
        if map_id not in self.visited_maps:
            reward += 1.0
        # The check `action == 0` requires that the agent use the 'a' button to trigger
        # the goal (e.g. to interact/confirm). Adjust if your goal should be action-agnostic.
        if full_pos in TARGET_POSITIONS and action == 0:
            reward += 10.0

        self.visited_maps.add(map_id)
        self.visited_positions.add(full_pos)
        self.last_pos = full_pos

        # Termination/truncation: we treat reaching the target as terminated;
        # reaching max gameplay time as truncated
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
        # Gymnasium reset returns (obs, info)
        return self.get_observation(), {}

    def close(self):
        try:
            self.pyboy.stop()
        except Exception:
            pass
