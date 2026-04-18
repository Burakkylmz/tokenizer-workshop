from __future__ import annotations

from pathlib import Path

from tokenizer_workshop.config import get_project_root, load_config


def read_text_file(path: str | Path) -> str:
    """
    Read a UTF-8 text file and return its content as a string.

    Educational note:
    We keep this helper small and explicit so learners can clearly see where
    raw training text enters the system.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Expected a file path, but got: {file_path}")

    return file_path.read_text(encoding="utf-8")


def get_sample_file_paths() -> list[Path]:
    """
    Return sample file paths defined in config.yaml.

    Paths in config.yaml are treated as project-relative paths.
    This keeps the config readable and avoids hardcoding absolute paths.
    """
    config = load_config()
    project_root = get_project_root()

    return [(project_root / path).resolve() for path in config.data.sample_files]


def load_sample_texts() -> dict[str, str]:
    """
    Load all sample text files declared in config.yaml.

    Returns:
        A dictionary where:
        - key   -> normalized project-relative file path (POSIX style)
        - value -> file content
    """
    texts: dict[str, str] = {}
    project_root = get_project_root()

    for path in get_sample_file_paths():
        relative_key = path.relative_to(project_root).as_posix()
        texts[relative_key] = read_text_file(path)

    return texts