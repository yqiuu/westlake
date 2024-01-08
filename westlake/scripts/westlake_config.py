import shutil
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--dirname', type=str, default='./', help='Directory to save the config file.')
    args = parser.parse_args()
    fname_target = Path(args.dirname)/Path("config.yml")

    fname = Path(__file__).parent.parent/Path("config")/Path("config_template.yml")
    shutil.copy(fname, fname_target)


if __name__ == "__main__":
    main()