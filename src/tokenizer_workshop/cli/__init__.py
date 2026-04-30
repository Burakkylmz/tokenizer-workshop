"""
CLI package

Tokenizer Workshop uygulamasının command-line interface katmanını içerir.

Bu katman:

- Kullanıcı etkileşimini yönetir (menu, input)
- Comparison pipeline'ını tetikler (runner)
- Uygulama akışını koordine eder (controller)

Modüller:

- menu.py        → CLI ekran çıktıları
- input.py       → kullanıcı input parsing
- runner.py      → comparison execution orchestration
- controller.py  → CLI flow orchestration

Not:
    Bu katman core tokenization logic içermez.
    Tokenization işlemleri CompareManager tarafından gerçekleştirilir.

Kullanım:

    from tokenizer_workshop.cli import CLIController, CLIComparisonRunner
"""

from .controller import CLIController
from .runner import CLIComparisonRunner

__all__ = [
    "CLIController",
    "CLIComparisonRunner",
]