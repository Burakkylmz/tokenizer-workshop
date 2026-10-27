from __future__ import annotations

import pytest

from tokenizer_workshop import main
from tokenizer_workshop.cli.controller import CLIController


def test_package_entrypoint_opens_menu_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Exercise the real entry point and controller without loading model backends.
    monkeypatch.setattr(CLIController, "_build_tokenizers", lambda self: {})
    monkeypatch.setattr("tokenizer_workshop.cli.controller.clear_screen", lambda: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "3")

    main()

    output = capsys.readouterr().out
    assert "TOKENIZER WORKSHOP CLI" in output
    assert "Exiting Tokenizer Workshop CLI." in output
