"""Run the FrontVeg V2 batch pipeline from a source checkout.

Equivalent to the ``frontveg`` command installed with the package:

    python main_final.py --input data/leaf --output results --prompt leaf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from frontveg.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
