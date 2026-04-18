from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str
    version: str
    debug: bool


@dataclass
class DataConfig:
    sample_files: list[str]


@dataclass
class TokenizerConfig:
    first_version: list[str]
    default_tokenizer: str


@dataclass
class LLMConfig:
    provider: str
    model: str
    use_for_explanations: bool
    api_key: str | None


@dataclass
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    tokenizers: TokenizerConfig
    llm: LLMConfig


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    root = get_project_root()
    full_path = root / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with full_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level dictionary/object.")

    return data


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    raw = load_yaml_config(config_path)

    project_raw = raw.get("project", {})
    data_raw = raw.get("data", {})
    tokenizers_raw = raw.get("tokenizers", {})
    llm_raw = raw.get("llm", {})

    project = ProjectConfig(
        name=project_raw.get("name", "tokenizer-workshop"),
        version=project_raw.get("version", "0.1.0"),
        debug=project_raw.get("debug", False),
    )

    data = DataConfig(
        sample_files=data_raw.get("sample_files", []),
    )

    tokenizers = TokenizerConfig(
        first_version=tokenizers_raw.get("first_version", []),
        default_tokenizer=tokenizers_raw.get("default_tokenizer", "char"),
    )

    llm = LLMConfig(
        provider=llm_raw.get("provider", "groq"),
        model=llm_raw.get("model", "llama-3.1-8b-instant"),
        use_for_explanations=llm_raw.get("use_for_explanations", False),
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return AppConfig(
        project=project,
        data=data,
        tokenizers=tokenizers,
        llm=llm,
    )