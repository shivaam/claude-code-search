import sys
from pathlib import Path

# Make src/ importable in tests without full install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
