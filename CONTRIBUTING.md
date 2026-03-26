# Contributing to AARS

Thank you for your interest in contributing to AARS!

## Development Setup

```bash
git clone https://github.com/lekhanpro/aars.git
cd aars
pip install -e ".[dev,ui]"
python -m spacy download en_core_web_sm  # optional
```

## Running Tests

```bash
pytest -q
python -m compileall src benchmarks tests
```

## Running Benchmarks

```bash
python benchmarks/runner.py --output benchmarks/results_local.json
```

## Code Style

- Python 3.11+ with type hints
- Use `structlog` for logging
- Follow existing patterns in `src/`
- All new modules need corresponding tests in `tests/`

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest -q`)
5. Commit with clear messages
6. Push and open a Pull Request

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- For benchmark issues, include `benchmarks/results_local.json`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
