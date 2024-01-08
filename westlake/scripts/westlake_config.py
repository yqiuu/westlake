import shutil
from pathlib import Path


def main():
    dirname = Path(__file__).parent.parent
    fname = dirname/Path("config")/Path("config_template.yml")
    shutil.copy(fname, "./config.yml")


if __name__ == "__main__":
    main()