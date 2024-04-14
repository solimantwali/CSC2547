import argparse
from pathlib import Path


def global_scores(file):
    pass


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=False, type=str)
args = parser.parse_args()

if __name__ == "__main__":
    file = Path(args.file)
    assert file.exists()
