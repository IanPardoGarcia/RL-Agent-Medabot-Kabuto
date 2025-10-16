from pathlib import Path
import sys
import argparse
import time
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MultiInputPolicy

# Ensure repo root is on sys.path so `import env...` works when running this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.generic_env import GenericPyBoyEnv
from pyboy import PyBoy


DATA_DIR = ROOT / "data"
ROM_PATH = DATA_DIR / "MedarotKabuto.gb"
STATE_PATH = DATA_DIR / "zero_state.state"
LOG_DIR = ROOT / "logs" / "ppo_medarot_v2"
TBOARD_DIR = ROOT / "logs" / "ppo_tensorboard_v2"
MODEL_DIR = ROOT / "models"


def make_env_fn(rom_path: Path = ROM_PATH, render: bool = False) -> Callable:
    def _init():
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}. Place the ROM in the data/ folder.")
        # Each environment creates its own PyBoy instance. Keep num_envs small by default.
        pyboy = PyBoy(str(rom_path))
        env = GenericPyBoyEnv(pyboy, debug=False, render_mode=None, state_path=STATE_PATH)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(LOG_DIR / "monitor.csv"))

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training v2 - remote-friendly defaults")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments (default: 1)")
    parser.add_argument("--use-subproc", action="store_true", help="Use SubprocVecEnv (heavier) instead of DummyVecEnv")
    parser.add_argument("--total-timesteps", type=int, default=10000, help="Total timesteps to train (default: 10k for quick tests)")
    parser.add_argument("--checkpoint-freq", type=int, default=5000, help="Save checkpoint every N steps (default: 5k)")
    parser.add_argument("--device", type=str, default="cpu", help="Training device: cpu or cuda (default: cpu)")
    parser.add_argument("--tensorboard-log", type=str, default=str(TBOARD_DIR), help="Tensorboard log dir")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Where to save models")
    parser.add_argument("--rom", type=str, default=str(ROM_PATH), help="Path to ROM file")
    parser.add_argument("--render", action="store_true", help="Enable render mode (slower)")
    parser.add_argument("--smoke", action="store_true", help="Run a single quick iteration and exit (for CI/smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()

    rom_path = Path(args.rom)
    num_envs = max(1, args.num_envs)

    make_env = make_env_fn(rom_path=rom_path, render=args.render)

    if args.use_subproc and num_envs > 1:
        env_fns = [make_env for _ in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        env_fns = [make_env for _ in range(num_envs)]
        vec_env = DummyVecEnv(env_fns)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TBOARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback saves intermediate models to allow resuming and remote-safe behavior
    checkpoint_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(MODEL_DIR), name_prefix="ppo_v2")

    # Build model with conservative defaults to avoid overheating user's hardware
    model = PPO(
        MultiInputPolicy,
        vec_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,
        tensorboard_log=args.tensorboard_log,
        device=args.device,
    )

    timesteps = args.total_timesteps
    if args.smoke:
        timesteps = 256

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=checkpoint_cb)
    elapsed = time.time() - start

    final_path = Path(args.model_dir) / "ppo_medarot_v2"
    model.save(str(final_path))

    print(f"Training finished. Trained for {timesteps} timesteps in {elapsed:.1f}s. Model saved to {final_path}")


if __name__ == "__main__":
    main()
