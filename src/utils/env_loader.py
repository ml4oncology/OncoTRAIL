"""
Environment loader for OncoTRAIL.

Loads cluster-specific environment variables from env.sh when running
Python scripts directly (outside of the SLURM workflow).
"""

import os
import subprocess
from pathlib import Path


def load_env(env_file: str = None) -> None:
    """
    Load environment variables from env.sh if not already set.

    This allows Python scripts to work both when called from shell scripts
    (which source env.sh) and when run directly.

    Parameters
    ----------
    env_file : str, optional
        Path to env.sh file. If None, looks for env.sh in the repository root.
    """
    if env_file is None:
        # Find env.sh relative to this file's location (src/utils/)
        # Go up to repo root: src/utils/ -> src/ -> repo_root/
        repo_root = Path(__file__).parent.parent.parent
        env_file = repo_root / "env.sh"

    env_file = Path(env_file)
    if not env_file.exists():
        return

    # Parse env.sh by sourcing it in a subprocess and capturing the exports
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {env_file} && env"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    # Only set if not already set (don't override existing env vars)
                    if key not in os.environ:
                        os.environ[key] = value
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
