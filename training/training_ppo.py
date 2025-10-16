from pathlib import Path

from pyboy import PyBoy
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from env.generic_env import GenericPyBoyEnv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ROM_PATH = DATA_DIR / "MedarotKabuto.gb"
STATE_PATH = DATA_DIR / "zero_state.state"
LOG_DIR = ROOT / "logs" / "ppo_medarot"
TBOARD_DIR = ROOT / "logs" / "ppo_tensorboard"
MODEL_DIR = ROOT / "models"


def make_env(rom_path: Path = ROM_PATH):
    def _init():
        # Each subprocess will create its own PyBoy instance. This is resource intensive
        # — ensure your system can handle `num_envs` separate emulator instances.
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}. Place the ROM in the data/ folder.")
        pyboy = PyBoy(str(rom_path))
        env = GenericPyBoyEnv(pyboy, debug=False, render_mode=False)
        # Keep only the 'info' array for the policy input
        env = TransformObservation(
            env,
            lambda obs: {"info": obs["info"]},
            observation_space=Dict({"info": Box(0, 255, (4,), dtype=np.float32)}),
        )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(LOG_DIR / "monitor.csv"))

    return _init


def main():
    num_envs = 4
    # SubprocVecEnv runs multiple independent envs in subprocesses. This helps
    # collect diverse rollouts in parallel but increases memory/cpu usage.
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE: hyperparameters below were chosen empirically; adjust as needed.
    model = PPO(
        MultiInputPolicy,
        vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.1,
        clip_range=0.2,
        tensorboard_log=str(TBOARD_DIR),
    )

    # Start learning — this can take a long time depending on timesteps and env speed.
    model.learn(total_timesteps=750_000)
    model.save(str(MODEL_DIR / "ppo_medarot"))


if __name__ == "__main__":
    main()
