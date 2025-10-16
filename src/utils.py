"""AXON Utilities

Common utility functions for the AXON pipeline.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path="axon.cfg.yml"):
    """Load AXON configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # Test utilities
    config = load_config()
    print(f"Loaded config for project: {config['project']}")