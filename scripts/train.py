"""Train a landmark network with the Hydra configs in configs/."""

import sys

from charm.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
