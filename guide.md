# Guide

This document is prepared for students who will contribute to the `tokenizer-workshop` project. The goal is to establish a common working standard for cloning the project locally, running it with `uv`, developing on a personal branch, pushing changes, and opening a Pull Request (PR).

The recommended working model for this repository is as follows:

* The main repository is the central workspace.
* The contributor **forks** the repository to their own account.
* Development is done on the contributor’s own fork, in their own **feature branch**.
* After completing the work, a **PR** is opened to the `main` branch of the main repository.

This approach both teaches a professional Git/GitHub workflow and keeps the main repository controlled.

---

## 1. Contribution approach

In this project, development is not done directly on the `main` branch.
Each student works on their own branch and opens a Pull Request to the `main` branch after completing their work.

The basic flow is:

1. Fork the repository
2. Clone your fork locally
3. Set up the project
4. Define the main repository as `upstream`
5. Create your branch
6. Do the development
7. Run the tests
8. Commit your changes
9. Push to your fork
10. Open a Pull Request to the main repository

---

## 2. Requirements

The main tools used in this project are:

* Git
* Python 3.10+
* `uv`

### Checking `uv`

In PowerShell:

```powershell
uv --version
```

If it returns a version, it is ready.

---

## 3. Cloning the repository locally

### Recommended method — Working with a fork

First, **fork** the main repository to your own account via GitHub.
Then clone your fork:

```powershell
git clone <YOUR_FORK_URL>
cd tokenizer-workshop
```

Example structure:

* Main repo: `https://github.com/Burakkylmz/tokenizer-workshop.git`
* Your fork: `https://github.com/<your-username>/tokenizer-workshop.git`

### Alternative method — If you have direct access

If you have direct write access to the repository, you can theoretically clone the main repo directly:

```powershell
git clone <REPO_URL>
cd tokenizer-workshop
```

However, for training and PR discipline, the recommended approach is still **fork + branch + PR** workflow.

---

## 4. Adding the main repository as `upstream`

If you are working with a fork, after cloning, `origin` points to your fork.
In this case, you should add the main repository as `upstream`:

```powershell
git remote add upstream https://github.com/Burakkylmz/tokenizer-workshop.git
```

To verify:

```powershell
git remote -v
```

Expected logic:

* `origin` -> your fork
* `upstream` -> main repository

This setup is important because it allows you to synchronize your fork as the main repository updates.

---

## 5. Setting up the project locally

After entering the repository directory, first synchronize dependencies:

```powershell
uv sync
```

This command:

* creates `.venv` (if it does not exist)
* installs dependencies according to `pyproject.toml` and `uv.lock`
* prepares the project environment

Then run the project entry point:

```powershell
uv run tokenizer-workshop
```

If this command runs successfully, the project is up and running at a basic level.

### Running tests

To run all tests:

```powershell
uv run pytest -v
```

To run a specific test file:

```powershell
uv run pytest tests/test_char_tokenizer.py -v
```

---

### Docker alternative for contributors

Python and uv are optional on the host when using Docker. Install Git and
Docker Desktop/Engine with Compose, then follow the [Docker setup in README](README.md#docker-web-cli-and-offline-tests).
The fork and feature-branch workflow remains the same.

```powershell
New-Item -ItemType Directory -Force results
docker compose build
docker compose run --rm tests
docker compose up -d --wait web
docker compose run --rm cli
docker compose down
```

Rebuild after editing source, tests, or YAML configuration: the image contains
these files and does not mount the repository. Tests run in the same Python 3.10
environment as the application, without networking or API credentials. Four
Hugging Face integration tests are opt-in (`-m integration`); see the README for
the network-enabled command. Local `uv run pytest` also defaults to offline tests.
Keep `.env` and generated `results/` out of commits. Review `git diff` and report
which tests actually ran; a successful image build alone is not runtime validation.

## 6. Pulling the latest code

Before starting work, update your local `main` branch.

### If working with a fork

First fetch the latest code from the main repository:

```powershell
git fetch upstream
git checkout main
git merge upstream/main
```

Then optionally push it to your fork:

```powershell
git push origin main
```

### If working directly with the main repository

```powershell
git checkout main
git pull origin main
```

---

## 7. Creating your branch

Each development should be done in a separate branch.
Branch names should be clear and concise.

Example branch names:

* `feature/word-tokenizer`
* `feature/regex-tokenizer`
* `feature/regex-bpe-tokenizer`
* `feature/byte-bpe-tokenizer`
* `test/metrics-improvements`
* `docs/contribution-guide`

To create and switch to a branch:

```powershell
git checkout -b feature/<your-work-name>
```

Example:

```powershell
git checkout -b feature/word-tokenizer
```

---

## 8. Development workflow standards

When contributing, follow these principles:

* Make small and controlled changes
* Do not add unnecessary files
* Add tests along with code
* Add comments if they improve educational value
* Do not break the existing folder structure
* Do not push directly to the `main` branch

### Basic checklist

Before pushing your code, check:

1. Does the project run?
2. Do the relevant tests pass?
3. Are new files in the correct directory?
4. Is `__init__.py` updated if necessary?
5. Are changes explainable and broken into small parts?

---

## 9. Checking file changes

To see the current status:

```powershell
git status
```

To see line-level changes:

```powershell
git diff
```

---

## 10. Committing changes

First stage files:

```powershell
git add .
```

For more controlled staging:

```powershell
git add src/tokenizer_workshop/tokenizers/word_tokenizer.py
git add tests/test_word_tokenizer.py
```

Then commit:

```powershell
git commit -m "Add word tokenizer and tests"
```

### Commit message suggestions

* `Add word tokenizer and tests`
* `Add regex tokenizer implementation`
* `Add byte BPE tokenizer draft`
* `Improve tokenizer metrics tests`
* `Update contribution guide`

Commit messages should be short, clear, and start with a verb.

---

## 11. Pushing to your fork

When pushing for the first time:

```powershell
git push -u origin feature/<your-work-name>
```

Example:

```powershell
git push -u origin feature/word-tokenizer
```

For subsequent pushes:

```powershell
git push
```

Important point:

* `origin` -> your fork
* push goes to your fork first
* no direct push to the main repository

---

## 12. Opening a Pull Request

After pushing, go to GitHub and open a PR for the relevant branch in your fork.

PR direction should be:

* **base repository:** main repo (`Burakkylmz/tokenizer-workshop`)
* **base branch:** `main`
* **compare branch:** your feature branch in your fork

### What to consider when opening a PR

The PR description should answer:

1. What was added or changed?
2. Why was this change made?
3. Which files were affected?
4. Which tests were run?

### Example PR template

```md
## Summary
This PR adds the first implementation of the tokenizer.

## Changes
- Added tokenizer implementation
- Added tests
- Updated package exports if needed

## Validation
- Ran `uv run pytest -v`
- Verified local run with `uv run tokenizer-workshop`

## Notes
- This PR focuses only on the current scope
- Follow-up improvements can be added separately
```

### Example PR titles

* `Add WordTokenizer implementation`
* `Add RegexTokenizer with tests`
* `Add ByteBPETokenizer baseline`
* `Improve tokenizer evaluation metrics`

---

## 13. Post-PR revision process

After opening a PR, comments may be received. In that case:

1. Make requested changes locally
2. Run tests again
3. Commit changes
4. Push to the same branch

Example:

```powershell
git add .
git commit -m "Address PR review comments"
git push
```

The same PR will be updated automatically.

---

## 14. Note on contributors visibility

For a contribution to appear in the main repository’s contributions:

* it must be merged into the default branch (e.g., `main`)

Just forking or committing to your fork is not enough.

Also, ensure correct Git configuration:

```powershell
git config user.name
git config user.email
```

To update:

```powershell
git config user.name "YourGitHubUsername"
git config user.email "your-email@example.com"
```

There may be a slight delay in GitHub reflecting contributions.

---

## 15. Frequently used commands

### Run the project

```powershell
uv run tokenizer-workshop
```

### Run all tests

```powershell
uv run pytest -v
```

### Run a single test file

```powershell
uv run pytest tests/test_simple_bpe_tokenizer.py -v
```

### Create a new branch

```powershell
git checkout -b feature/<your-work-name>
```

### Check status

```powershell
git status
```

### Commit

```powershell
git add .
git commit -m "Your commit message"
```

### Push

```powershell
git push -u origin feature/<your-work-name>
```

---

## 16. Mistakes to avoid when contributing

* Working on the `main` branch
* Pushing without running tests
* Opening very large and messy PRs
* Solving multiple unrelated topics in a single PR
* Making unnecessary refactors
* Breaking naming or folder structure
* Trying to merge non-working “draft” code into main
* Forgetting to open a PR after pushing to fork

---

## 17. Minimum expected contribution quality

For a contribution to be acceptable:

* Code must run locally
* Relevant tests must be written
* Must be compatible with the existing structure
* PR description must be clear
* Scope of change must be understandable

---

## 18. Final recommendation

The goal of this project is not only to write code, but also to explain what is written.
Therefore, when contributing, you should be able to clearly answer:

**"What did I do, why did I do it, and how did I validate it?"**

This is the most important point in PR evaluation.

