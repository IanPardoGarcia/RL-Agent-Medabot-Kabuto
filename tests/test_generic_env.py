from env.generic_env import GenericPyBoyEnv


class DummyPyBoy:
    """Minimal PyBoy-like object for unit tests: exposes memory, button, tick, load_state, stop."""

    def __init__(self):
        # initialize memory with zeros and set some sensible defaults
        self.memory = bytearray(0xFFFF)
        # place default player pos: x=16, y=8, map=80, orientation=1
        self.memory[0xC0D4] = 16
        self.memory[0xC0D5] = 8
        self.memory[0xC92D] = 80
        self.memory[0xC0D8] = 1

    def button(self, name):
        # no-op for tests
        pass

    def tick(self):
        # no-op for tests
        pass

    def load_state(self, f):
        # pretend to load - no-op
        return True

    def stop(self):
        pass


def test_generic_env_observation_and_step(tmp_path):
    dummy = DummyPyBoy()
    # create a dummy state file so GenericPyBoyEnv doesn't raise
    state_file = tmp_path / "zero_state.state"
    state_file.write_bytes(b"state")

    env = GenericPyBoyEnv(dummy, debug=True, render_mode=False, state_path=state_file)

    obs, info = env.reset()
    assert "info" in obs
    arr = obs["info"]
    assert arr.shape == (4,)
    assert arr[0] == 16

    # step with first action (index 0 -> ACTIONS[0] == 'a')
    next_obs, reward, terminated, truncated, _ = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()
