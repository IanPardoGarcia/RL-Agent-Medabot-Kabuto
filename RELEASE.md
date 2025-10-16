Release v1.0.0 - RL-Agent-Medarot-Kabuto

Summary:
- Initial working implementation of a Gymnasium environment wrapping PyBoy (GenericPyBoyEnv).
- Training scripts for PPO (import-safe) and basic evaluation script.
- Basic smoke tests for environment and training loop (mocked), plus a visualizer for trajectories.
- Lightweight CI workflow that runs smoke tests and lint checks.

Included in v1:
- `env/generic_env.py` - Gym env with Gymnasium API-compatible returns.
- `training/training_ppo.py` - training entrypoint (safe imports, project-relative paths).
- `evaluation/visualize_trajectory.py` - simple plotter for trajectory CSVs.
- `tests/` - smoke tests for environment and training loop.
- `LICENSE` (MIT), `CONTRIBUTING.md`, `requirements-dev.txt`.

Known limitations:
- End-to-end training requires the ROM `MedarotKabuto.gb` and saved state `zero_state.state` in `data/`.
- PyBoy and training dependencies are heavy and not installed in CI by default.
- Limited unit test coverage; more tests recommended for env edge cases and training integration.

Suggested tag command (local):

```
# create tag and push
git tag -a v1.0.0 -m "v1.0.0: initial public release"
git push origin v1.0.0
```
