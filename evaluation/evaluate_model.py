from pyboy import PyBoy
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Box, Dict
import numpy as np
from env.generic_env import GenericPyBoyEnv

model = PPO.load("ppo_medarot")

def make_env():
    pyboy = PyBoy("MedarotKabuto.gb")
    env = GenericPyBoyEnv(pyboy, debug=False, render_mode=False)
    env = TransformObservation(env,
        lambda obs: {"info": obs["info"]},
        observation_space=Dict({"info": Box(0, 255, (4,), dtype=np.float32)})
    )
    return Monitor(env)

num_episodes = 100
rewards = []
eval_env = make_env()

for ep in range(num_episodes):
    obs, _ = eval_env.reset()
    total = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total += reward
    rewards.append(total)
    print(f"Episodio {ep+1}: {total}")

print("Recompensa promedio:", np.mean(rewards))
