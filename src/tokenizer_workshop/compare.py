from __future__ import annotations

from tokenizer_workshop.comparisons.report import print_all_sample_results
from tokenizer_workshop.comparisons.runner import TokenizerComparator


def main() -> None:
    """
    Projedeki sample text'ler üzerinde varsayılan tokenizer compare akışını çalıştırır.
    """
    comparator = TokenizerComparator()
    all_results = comparator.run_all_samples()
    print_all_sample_results(all_results)


if __name__ == "__main__":
    main()


# For Run:
# uv run python -m tokenizer_workshop.compare