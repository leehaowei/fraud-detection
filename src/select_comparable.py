# import self-defined packages
from comparable_selector import ComparableSelector

# Constant
MODE = "local"

PERCENTILE_LOWER_BOUND = 0.8
PERCENTILE_UPPER_BOUND = 1.2
TARGET_PERCENTILE_COLUMN = "at"


def main():
    cp = ComparableSelector(
        n_records=10,
        column_compared=TARGET_PERCENTILE_COLUMN,
        lower_bound=PERCENTILE_LOWER_BOUND,
        upper_bound=PERCENTILE_UPPER_BOUND,
        mode=MODE,
    )
    cp.process()


if __name__ == "__main__":
    main()
