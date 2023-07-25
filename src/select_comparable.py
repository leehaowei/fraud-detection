# import self-defined packages
from preprocess.comparable_selector import ComparableSelector

# Constant
from config.comparable_config import ComparableConfig


def main():
    config = ComparableConfig()
    cp = ComparableSelector(config)
    cp.process()


if __name__ == "__main__":
    main()
