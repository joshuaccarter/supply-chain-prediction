"""Functions for reading and validating YAML configuration files.

This module resolves repository root paths and loads runtime configurations, including
data filepaths, column schemas, preprocessing drop lists, cross-validation parameters,
and LightGBM quantile regression settings.
"""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and parse the project YAML configuration file into a nested dictionary.

    Reads the specified YAML file from disk, verifying its existence and safely
    parsing its contents using `yaml.safe_load`.

    Args:
        config_path: Path to the YAML configuration file. Defaults to DEFAULT_CONFIG_PATH
            (`config/config.yaml` located at the repository root).

    Returns:
        dict[str, Any]: Dictionary containing all parsed top-level sections (e.g. `data`,
            `preprocessing`, and `models`).

    Raises:
        FileNotFoundError: If the specified `config_path` does not exist on disk.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
