"""Repository locations shared by the ml scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
FIXTURES_DIR = REPO_ROOT / "ml" / "fixtures"
EXPORT_RECORD_PATH = OUTPUTS_DIR / "exports" / "pretrained_yolo26n.json"
