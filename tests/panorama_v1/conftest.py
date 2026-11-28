from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PANORAMA = ROOT / "colab" / "real_scenes" / "panorama_v1"
sys.path.insert(0, str(PANORAMA))
