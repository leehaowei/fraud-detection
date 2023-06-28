# import self-defined packages
from comparable_preprocessor import ComparablePreprocessor

# Constant
MODE = "local"

PERCENTILE_LOWER_BOUND = 0.8
PERCENTILE_UPPER_BOUND = 1.2


def main():
    cp = ComparablePreprocessor(
        n_records=10,
        lower_bound=PERCENTILE_LOWER_BOUND,
        upper_bound=PERCENTILE_UPPER_BOUND,
    )
    cp.process()


if __name__ == "__main__":
    main()
