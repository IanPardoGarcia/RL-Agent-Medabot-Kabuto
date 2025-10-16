import subprocess
import sys
from pathlib import Path


def test_smoke_training_v2_runs():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "training" / "training_ppo_v2.py"
    assert script.exists(), "training_ppo_v2.py not found"

    # Run the script in smoke mode which should be quick and low-resource
    res = subprocess.run([sys.executable, str(script), "--smoke"], capture_output=True, text=True, timeout=60)
    print(res.stdout)
    print(res.stderr)
    assert res.returncode == 0, f"Script exited with {res.returncode}: {res.stderr}"
