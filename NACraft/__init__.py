"""NACraft package bootstrap.

The project keeps executable modules and bundled dependencies under the same
inner directory so scripts can also run from `cd NACraft && python main.py`.
Expose the same paths when imported as a package.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "boltz" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
