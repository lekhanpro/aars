# Contributing to AARS

Thank you for your interest in contributing to AARS! This guide will help you get started.

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aars.git
cd aars

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install all development dependencies
pip install -e ".[dev,bench,ui]"

# 4. Download the spaCy NER model
python -m spacy download en_core_web_sm

# 5. Start ChromaDB for local development
docker run -d -p 8001:8000 chromadb/chroma:latest

# 6. Copy the environment template
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Running Tests

```bash
# Unit tests only (fast, no external dependencies)
pytest tests/unit/ -v

# Full test suite (requires ChromaDB running)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The configuration is in `pyproject.toml`.

### Rules enforced:
- **Line length**: 100 characters maximum
- **Import sorting**: isort-compatible (I)
- **Naming conventions**: PEP 8 (N)
- **Pyflakes**: Error detection (F)
- **pycodestyle**: Style violations (E, W)
- **pyupgrade**: Modern Python syntax (UP)

### Check and fix:

```bash
# Check for lint violations
ruff check src/ tests/

# Auto-fix fixable violations
ruff check --fix src/ tests/

# Check formatting
ruff format --check src/ tests/

# Apply formatting
ruff format src/ tests/
```

All PRs must pass both `ruff check` and `ruff format --check` in CI.

## Submitting a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Make your changes and write tests for new functionality.

3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

4. Ensure code style is clean:
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

5. Commit with a clear message following conventional commits:
   ```bash
   git commit -m "feat: add support for custom retrieval strategies"
   ```

6. Push and open a PR against `main`.

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Adding or updating tests
- `refactor:` — Code change that neither fixes a bug nor adds a feature
- `chore:` — Maintenance tasks

## Architecture Overview

Before contributing, familiarize yourself with the project structure:

- **`config/`** — All configuration via Pydantic BaseSettings
- **`src/agents/`** — LLM-based planner and reflection agents
- **`src/retrieval/`** — Retriever implementations (vector, keyword, graph)
- **`src/fusion/`** — Result merging (RRF) and diversity reranking (MMR)
- **`src/pipeline/`** — Core orchestration logic
- **`src/api/`** — FastAPI endpoints and Pydantic schemas
- **`benchmarks/`** — Evaluation framework

## Reporting Issues

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for bugs
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
