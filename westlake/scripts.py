from pathlib import Path
from argparse import ArgumentParser

from .config import save_config_template


def exec_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--dir', type=str, default='./', help='Directory to save the config file.')
    args = parser.parse_args()
    save_config_template(Path(args.dir)/Path("config.yml"))