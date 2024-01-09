import shutil
from pathlib import Path
from argparse import ArgumentParser

from ..config import save_config_template


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--dirname', type=str, default='./', help='Directory to save the config file.')
    args = parser.parse_args()
    save_config_template(Path(args.dirname)/Path("config.yml"))


if __name__ == "__main__":
    main()