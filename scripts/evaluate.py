"""Score a checkpoint on the validation or test split."""

import sys

from charm.cli.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
