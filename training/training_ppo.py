from pyboy import PyBoy
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from env.generic_env import GenericPyBoyEnv

def make_env():
    def _init():
        pyboy = PyBoy("MedarotKabuto.gb")
        env = GenericPyBoyEnv(pyboy, debug=False, render_mode=False)
        env = TransformObservation(env,
                                   lambda obs: {"info": obs["info"]},
                                   observation_space=Dict({"info": Box(0, 255, (4,), dtype=np.float32)}))
        return Monitor(env, filename="./ppo_logs")
    return _init

vec_env = SubprocVecEnv([make_env() for _ in range(8)])

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
    tensorboard_log="./ppo_tensorboard/"
)

model.learn(total_timesteps=750_000)
model.save("ppo_medarot")
