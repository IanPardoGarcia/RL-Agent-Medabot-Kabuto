import argparse
from pyboy import PyBoy
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Box, Dict
import numpy as np
from env.generic_env import GenericPyBoyEnv


def make_env(rom_path: str = "MedarotKabuto.gb"):
    pyboy = PyBoy(rom_path)
    env = GenericPyBoyEnv(pyboy, debug=False, render_mode=False)
    env = TransformObservation(env,
        lambda obs: {"info": obs["info"]},
        observation_space=Dict({"info": Box(0, 255, (4,), dtype=np.float32)})
    )
    return Monitor(env)


def evaluate(model_path: str = "ppo_medarot", rom_path: str = "MedarotKabuto.gb", num_episodes: int = 100):
    model = PPO.load(model_path)
    env = make_env(rom_path)

    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        rewards.append(total)
        print(f"Episode {ep+1}: {total}")

    print("Average reward:", np.mean(rewards))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_medarot",
                        help="Path to trained model")
    parser.add_argument("--rom", default="MedarotKabuto.gb",
                        help="Path to ROM file")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run")
    args = parser.parse_args()
    evaluate(model_path=args.model, rom_path=args.rom, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
