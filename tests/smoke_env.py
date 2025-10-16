"""Simple smoke test that exercises GenericPyBoyEnv using a DummyPyBoy (no ROM required).
Run: python tests/smoke_env.py

Note: This is a lightweight manual smoke test. In CI we run a similar script but avoid
heavy dependencies like PyBoy; this test uses a DummyPyBoy and a temporary state file.
"""
import sys
from pathlib import Path
from pathlib import Path as _Path

# ensure project root is importable when running this script from tests/
PROJECT_ROOT = str(_Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
class DummyPyBoy:
    def __init__(self):
        self.memory = bytearray(0xFFFF)
        self.memory[0xC0D4] = 16
        self.memory[0xC0D5] = 8
        self.memory[0xC92D] = 80
        self.memory[0xC0D8] = 1

    def button(self, name):
        pass

    def tick(self):
        pass

    def load_state(self, f):
        return True

    def stop(self):
        pass


def main():
    # create a small dummy state file so GenericPyBoyEnv can load it
    tests_dir = Path(__file__).resolve().parent
    state_file = tests_dir / "dummy_zero_state.state"
    state_file.write_bytes(b"dummy")

    # import env after ensuring project root is on sys.path
    from env.generic_env import GenericPyBoyEnv

    dummy = DummyPyBoy()
    env = GenericPyBoyEnv(dummy, debug=True, render_mode=False, state_path=state_file)

    print('Resetting env...')
    obs, info = env.reset()
    print('Initial obs:', obs)

    print('Stepping 5 times...')
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        print(f'step {i+1}: reward={reward}, terminated={terminated}, truncated={truncated}, obs={obs}')
        if terminated or truncated:
            break

    env.close()
    print('Smoke env test completed successfully.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Smoke env test failed:', e)
        sys.exit(1)
    sys.exit(0)
