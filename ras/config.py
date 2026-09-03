"""
RAS Project Configuration Module.
Centralizes default paths, system constants, physics parameters, and model locations.
"""

from pathlib import Path

# Base Directory (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Asset Directories & Paths
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = ASSETS_DIR / "docs"
DEFAULT_DXF_PATH = ASSETS_DIR / "map_dxf.dxf"

# RL Model Directories & Paths
MODELS_DIR = BASE_DIR / "rl_models"
DEFAULT_RL_MODEL_PATH = MODELS_DIR / "best_model.zip"

# Output Directories
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Simulation & Physics Constants
DT = 0.002             # MuJoCo physics timestep (500 Hz)
PHYSICS_STEPS_PER_CONTROL = 10  # Control loop runs at 50 Hz (0.02s per step)
MAX_LINEAR_VEL = 1.0   # m/s
MAX_ANGULAR_VEL = 1.0  # rad/s
