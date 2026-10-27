# tokenizer-workshop

## Project Overview

`tokenizer-workshop` is a Python project developed to teach the concept of tokenization in a practical and education-focused manner.

The main goal of this project is to provide a strong answer to the following fundamental question by building different tokenizer approaches step by step:

**How is a piece of text split into meaningful parts by a machine, and why is this split important?**

This repository aims not only to use existing tokenizer libraries, but also to understand the logic underlying them.

## Problem Definition

### Problem

Tokenization is one of the most fundamental layers of NLP and LLM systems; however, in most trainings it is either explained superficially or passed over in a black-box manner through ready-made libraries.

As a result, learners generally:

* give incomplete answers to the question of what a token is,
* have difficulty distinguishing between character-level, byte-level, and subword approaches,
* cannot fully understand why methods like BPE emerged,
* cannot explain why the same text produces different token counts across different tokenizers.

### Why this problem matters

Many critical topics such as LLMs, fine-tuning, embeddings, context window, cost, and latency are directly related to tokenization. Without understanding tokenization logic, it is difficult to develop a deep understanding of LLM system design.

### Target user / use case

This project is suitable for the following users:

* developers learning AI / NLP
* engineers who want to understand LLM systems more deeply
* instructors who provide technical training
* students who want to learn tokenization by writing code

## Solution Approach

This project teaches the concept of tokenization not through a single class, but in a comparative and incremental manner.

The following approaches are mainly covered in the project:

* `CharTokenizer`
* `ByteTokenizer`
* `SimpleBPETokenizer`

In this way, the learner can clearly see the following progression:

* character-level representation
* byte-level representation
* subword / merge-based representation

### Architecture summary

The project consists of a Python package developed under a clean `src/` structure. Application settings are stored in `config.yaml`, while project metadata and dependency information are stored in `pyproject.toml`. The development workflow is managed with `uv`. Secret values are not written into the repository; when necessary, they are read via environment variables. The goal is not to chase production-grade performance, but to create a tokenizer laboratory that is readable, testable, and suitable for teaching.

## Tech Stack

| Component                | Choice                | Notes                                     |
| ------------------------ | --------------------- | ----------------------------------------- |
| Language                 | Python                | Python 3.10+                              |
| Environment & workflow   | uv                    | Dependency and environment management     |
| Project metadata         | pyproject.toml        | Central package and dependency management |
| App config               | YAML                  | Application settings via `config.yaml`    |
| Tokenizer implementation | Custom                | Education-focused custom implementation   |
| UI / Interface           | CLI / script          | Simple usage                              |
| Evaluation               | Simple custom metrics | token count, vocab size, comparison       |

## Project Structure

```text
src/
└── tokenizer_workshop/
    ├── tokenizers/
    ├── trainers/
    ├── evaluators/
    └── utils/

tests/
data/
README.md
config.yaml
pyproject.toml
```

### Folder descriptions

* `src/tokenizer_workshop/`: Main application code
* `tests/`: Test files
* `data/`: Sample texts and small demo inputs

## Setup

### Requirements

* Python version: **3.10+**
* Required tool: **uv**
* Optional secret: **GROQ_API_KEY**

### Installation

```bash
uv sync
```

### Environment Variables

If needed, the following value can be defined via system environment variables:

```env
GROQ_API_KEY=
```

Note: API key values must not be written into the repository.

## Run Instructions

To run the project entry point:

```bash
uv run tokenizer-workshop
```

To run the tests:

```bash
uv run pytest -v
```

## Docker: Web, CLI, and offline tests

### Prerequisites and first run

Install Git and Docker Desktop (Windows/macOS) or Docker Engine with the Compose
plugin (Linux). On Docker Desktop, select Linux containers and start the engine.
You do **not** need Python or uv installed on your computer. Internet access is
needed for the first image build and optional Hugging Face downloads.

Run in PowerShell (the Docker commands also work in Bash):

```powershell
git clone https://github.com/Burakkylmz/tokenizer-workshop.git
cd tokenizer-workshop
# Optional: only needed to change the port or Linux user IDs.
Copy-Item .env.example .env
New-Item -ItemType Directory -Force results
docker compose build
docker compose up -d --wait web
```

On Linux/macOS use `cp .env.example .env` and `mkdir -p results` instead of the
two PowerShell filesystem commands. On Linux, set `WORKSHOP_UID` and
`WORKSHOP_GID` in `.env` to the values printed by `id -u` and `id -g` so CLI
reports belong to your host user. Create `results/` before running the CLI.

Open:

- Web UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/api/health
- OpenAPI schema: http://localhost:8000/openapi.json

Start with `char`, `byte`, or `word` tokenizers. These require no credentials,
model downloads, or internet at runtime. The server listens on `0.0.0.0:8000`
inside the container; Compose exposes it only on the host loopback interface.
Set `WORKSHOP_PORT=8001` in `.env` to use a different host port, then run
`docker compose up -d --wait web` again and open `http://localhost:8001`.

### CLI and persistent reports

```powershell
docker compose run --rm cli
```

Choose `1` for the built-in text, then `1,2,3` for word/character/byte comparison.
Press Enter after the report, then `3` to exit. Choose `2` to enter custom text.
Each comparison creates a distinct `results/report-*.md` on your computer;
files survive `--rm` and `docker compose down`. Existing reports are preserved.
Local `uv run tokenizer-workshop` retains its existing `report.md` behavior.
Set `TOKENIZER_WORKSHOP_OUTPUT_DIR` locally if you also want unique files there.

The current CLI has built-in/custom text input, not a file-picker menu. Sample
files are packaged under `/app/data` and can be read with the existing helper:

```powershell
docker compose run --rm cli python -c "from tokenizer_workshop.utils.data_loader import load_sample_texts; print(load_sample_texts())"
Get-ChildItem results
```

Web Markdown and PDF downloads are saved by your browser. They are not CLI
reports and do not need a server volume. PDF response files use unique temporary
paths and are deleted after delivery (or if generation fails).

### Testing, logs, and shutdown

```powershell
# Runs without network access or API credentials.
docker compose run --rm tests
docker compose logs --tail=100 web
docker compose ps
# Stop/remove containers; keep host reports.
docker compose down
# After changing source, tests, config.yaml, or dependencies:
docker compose build
docker compose up -d --wait web
```

`pytest.ini` excludes four existing `integration` tests that load a Hugging Face
tokenizer. Mock-based pretrained tests remain in the offline suite. The suite
also covers Web assets, API requests, complete CLI comparisons, sample data,
Markdown reports, ReportLab PDFs, PDF cleanup, and Linux WeasyPrint libraries.
To deliberately run model-download integration tests with network access:

```powershell
docker compose run --rm cli pytest -q -p no:cacheprovider -m integration
```

The `pretrained` tokenizer (including CLI selection `all` or an empty tokenizer
or invalid selection) may download a model
on first use. Its cache is disposable in this setup and may be downloaded again
in a new container. Select local tokenizers for disconnected workshops. No models
are downloaded during builds or offline tests. `config.yaml` remains the YAML
configuration source; edit it and rebuild. The CLI's tokenizer settings remain
in `compare.py`, as before.

No Groq API calls exist in the current application paths. `GROQ_API_KEY` is
optional configuration metadata, read only when `llm.use_for_explanations` is
enabled. Compose does not inject it. `.env` is only used for the documented
Compose variables, never copied into the image; never commit credentials.

### Image design and troubleshooting

One image serves all three workflows. `web` is the default service; `cli` and
`tests` are opt-in tools sharing its installed environment. The image uses Python
3.10, a version-pinned uv, the existing unchanged lockfile, and frozen dependency
installation. Commands invoke installed executables directly, so startup does
not resolve or install dependencies. The image runs as a non-root user and
contains the source, YAML, samples, static assets, tokenizer guides, and tests.
Only CLI reports are bind-mounted; your source and host virtual environment are
not mounted. Restart is manual to keep training failures visible.

The single-stage build avoids duplicating environments; no compiler toolchain
is needed for the locked Linux wheels. Dependency layers are cached separately
from source changes, following the [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
Pango, Harfbuzz and fonts are included for native PDF support, following the
[WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/v68.1/first_steps.html).
The application PDF endpoint uses ReportLab; WeasyPrint is also smoke-tested
because it is a declared dependency. Versioned base-image tags and Debian package
repositories can still change: frozen Python dependencies do not imply a
byte-for-byte identical OS image across future rebuilds.

| Problem | Action |
| --- | --- |
| Cannot connect to Docker | Start Docker Desktop; run `docker version` and confirm a Server section. Use Linux containers. |
| Port 8000 occupied | Set `WORKSHOP_PORT=8001` in `.env`, recreate Web, and use port 8001 in your browser. |
| Build/download failure | Check registry/PyPI/Debian connectivity and Docker proxy settings. Retry `docker compose build --progress=plain`. Do not delete or regenerate the lockfile to bypass an error. |
| Missing native PDF libraries | Rebuild the provided Dockerfile; inspect `docker compose logs web`. Run `docker compose run --rm cli python -m weasyprint --info`. |
| Configuration ignored | `.env` controls Compose interpolation only; YAML/source edits require a rebuild. Verify variable names against `.env.example`. |
| CLI input/TTY error | Run `docker compose run --rm cli` in a terminal. For piped input use `docker compose run --rm -T cli`. |
| Report permission denied | Create `results/` as your host user. On Linux set UID/GID as above; on Docker Desktop check file sharing permissions. Do not use privileged mode. |
| Hugging Face failure offline | Avoid `pretrained` and `all`; use local tokenizers. Optional integration tests need network access or cached models. |
| PDF/report failure | Check logs and rerun `docker compose run --rm tests`; browser downloads and CLI `results/` are separate output locations. |
| Code changes not visible | Rebuild and recreate the Web service; there is no automatic source mount/reload. |
| Custom CLI text has unknown characters | Some educational tokenizers reject characters absent from the built-in training text. Use the default comparison for the smoke test, or select `byte` for arbitrary custom text. This is existing tokenizer behavior. |

## Example Input / Output

### Example input

```text
Merhaba dünya!
```

### Example output

```text
CharTokenizer -> character-level tokens  
ByteTokenizer -> UTF-8 byte ids  
SimpleBPETokenizer -> learned subword tokens  
```

## Key Features

* Education-focused tokenizer design
* Comparative learning approach
* Showing character, byte, and BPE levels together
* Test-driven development workflow
* Simple but instructive metrics

## Limitations

This project has the following limitations:

* does not aim for production-grade tokenizer performance
* does not solve large-scale data and optimization problems
* does not aim to perfectly replicate all real-world tokenizer behaviors

## Future Improvements

The following improvements can be made in the future:

* adding `WordTokenizer`
* adding `RegexTokenizer`
* adding `RegexBPETokenizer`
* adding `ByteBPETokenizer`
* merge trace / visualization module
* notebook-based training materials

## Project Status

**Status:** in progress

## Repository Workflow

* Development progresses in a controlled and step-by-step manner.
* Large bulk changes are avoided.

## Author

* Name: Burak
* Project Topic: Educational tokenizer workshop for learning tokenization step by step

