"""Smoke test that runs a tiny PPO training on a mock environment to verify training loop works.
Run: python tests/smoke_training.py

This test is intended for local development. It will skip automatically if
`stable-baselines3` is not installed (so CI can avoid installing heavy ML frameworks).
"""
import sys
import numpy as np
from pathlib import Path as _Path

# ensure project root is importable when running from tests/
PROJECT_ROOT = str(_Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # import heavy/third-party modules after ensuring project root is on sys.path
    from gymnasium import Env
    from gymnasium.spaces import Discrete, Box

    class DummyEnv(Env):
        def __init__(self):
            super().__init__()
            self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
            self.action_space = Discrete(2)
            self._step = 0

        def reset(self, seed=None, **kwargs):
            self._step = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self._step += 1
            obs = np.random.rand(4).astype(np.float32)
            reward = 1.0
            terminated = self._step >= 5
            truncated = False
            return obs, float(reward), terminated, truncated, {}

        def close(self):
            pass

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except Exception:
        print('stable-baselines3 not installed; skipping smoke training test.')
        return

    env = make_vec_env(DummyEnv, n_envs=1)
    model = PPO('MlpPolicy', env, verbose=0)
    print('Starting short learn() (200 steps)...')
    model.learn(total_timesteps=200)
    print('Finished learn()')
    reset_res = env.reset()
    # make_vec_env returns a VecEnv; reset may return just observations or (obs, info)
    if isinstance(reset_res, tuple) and len(reset_res) == 2:
        obs, _info = reset_res
    else:
        obs = reset_res

    # model.predict may return only action when deterministic or single env
    pred = model.predict(obs)
    if isinstance(pred, tuple) and len(pred) >= 1:
        action = pred[0]
    else:
        action = pred

    print('Sample prediction OK, action:', action)
    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Smoke training failed:', e)
        sys.exit(1)
    sys.exit(0)
