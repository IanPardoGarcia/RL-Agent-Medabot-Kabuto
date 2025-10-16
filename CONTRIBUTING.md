Contributing
============

Thanks for your interest in contributing to RL-Agent-Medarot-Kabuto!

Getting started
---------------
1. Fork the repository and create a feature branch.
2. Install dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run tests:

```powershell
# run unit/smoke tests
python -c "import runpy; runpy.run_path('tests/test_generic_env.py', run_name='__main__')"
python -c "import runpy; runpy.run_path('tests/smoke_env.py', run_name='__main__')"
```

Coding style
------------
- We follow a lightweight style: please run `black` and `flake8` before submitting a PR.
- Type hints are encouraged; run `mypy` to check typing.

Opening a PR
------------
- Keep changes small and focused.
- Include tests for any non-trivial logic.
- Describe how to reproduce the issue and how your change fixes it.

Maintainers
-----------
Ian Pardo Garcia
