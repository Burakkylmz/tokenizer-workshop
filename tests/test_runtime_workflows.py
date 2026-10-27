"""Offline workflow coverage shared by local development and Docker."""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from tokenizer_workshop.api.main import app
from tokenizer_workshop.api.routes import tokenize as routes
from tokenizer_workshop.cli.controller import CLIController
from tokenizer_workshop.config import load_config
from tokenizer_workshop.utils.data_loader import load_sample_texts


def test_menu_does_not_construct_unused_tokenizers(monkeypatch):
    calls = []
    selected = object()

    def create(name, **kwargs):
        assert name == "char", "Unselected tokenizer was constructed"
        calls.append(name)
        return selected

    monkeypatch.setattr(routes.TokenizerFactory, "create", create)
    controller = CLIController(None, {"char": {}, "pretrained": {}}, "hello")
    assert calls == []
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert controller._select_tokenizers() == {"char": selected}
    assert controller._select_tokenizers() == {"char": selected}
    assert calls == ["char"]


def test_cli_comparisons_persist_unique_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZER_WORKSHOP_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    completed = subprocess.run(
        [sys.executable, "-c", "from tokenizer_workshop import main; main()"],
        input="1\n1,2,3\n\n2\nHello world!\n2,3\n\n3\n",
        text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Exiting Tokenizer Workshop CLI" in completed.stdout
    reports = list(tmp_path.glob("report-*.md"))
    assert len(reports) == 2
    assert all(path.stat().st_size > 100 for path in reports)


def test_local_report_path_is_preserved(tmp_path, monkeypatch):
    from tokenizer_workshop.cli.runner import CLIComparisonRunner
    from tokenizer_workshop.comparisons.compare_manager import CompareManager
    from tokenizer_workshop.tokenizers.char_tokenizer import CharTokenizer

    monkeypatch.delenv("TOKENIZER_WORKSHOP_OUTPUT_DIR", raising=False)
    report = tmp_path / "local.md"
    CLIComparisonRunner(CompareManager(), "hello", str(report)).run(
        "hello", {"char": CharTokenizer()}
    )
    assert report.stat().st_size > 100


def test_config_and_samples_without_credentials(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-test-value")
    assert load_config().llm.api_key is None
    samples = load_sample_texts()
    assert len(samples) == 3
    assert all(samples.values())


def test_web_assets_tokenization_and_markdown():
    with TestClient(app) as client:
        for path in ("/api/health", "/", "/css/style.css", "/js/app.js",
                     "/docs", "/openapi.json", "/api/tokenizers"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.content
        response = client.post("/api/tokenize", json={
            "text": "hello", "tokenizer_name": "char",
        })
        assert response.status_code == 200, response.text
        assert response.json()["token_count"] == 5
        assert len(response.json()["tokens"]) == 5
        response = client.post("/api/report", json={
            "text": "hello world", "tokenizer_names": ["char", "byte"],
            "format": "md",
        })
        assert response.status_code == 200, response.text
        assert response.json()["format"] == "md"
        assert len(response.json()["report"]) > 100


def test_pdf_response_has_unique_files_and_cleans_up(monkeypatch):
    paths = []
    original = routes.build_pdf_report

    def record(source, path):
        paths.append(path)
        return original(source, path)

    monkeypatch.setattr(routes, "build_pdf_report", record)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/api/report/pdf", json={
                "text": "hello world", "tokenizer_names": ["char", "byte"],
            })
            assert response.status_code == 200, response.text
            assert response.content.startswith(b"%PDF-")
            assert len(response.content) > 1000
    assert paths[0] != paths[1]
    assert all(not path.exists() for path in paths)


def test_failed_pdf_generation_cleans_up(monkeypatch):
    paths = []

    def fail(source, path):
        paths.append(path)
        raise RuntimeError("Deliberate rendering failure")

    monkeypatch.setattr(routes, "build_pdf_report", fail)
    with TestClient(app) as client:
        response = client.post("/api/report/pdf", json={
            "text": "hello", "tokenizer_names": ["char"],
        })
    assert response.status_code == 500
    assert paths and not paths[0].exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native-library smoke test")
def test_weasyprint_native_libraries():
    from weasyprint import HTML

    pdf = HTML(string="<h1>Tokenizer workshop</h1><p>Native library check</p>").write_pdf()
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000
