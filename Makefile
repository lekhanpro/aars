.PHONY: install dev test lint run docker-up docker-down bench paper ui

install:
	pip install -e .

dev:
	pip install -e ".[dev,bench,ui]"

test:
	pytest tests/unit/ -v --tb=short

test-all:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

bench:
	python -m benchmarks.runner --datasets hotpotqa nq triviaqa msmarco

paper:
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

ui:
	streamlit run ui/app.py --server.port 8501
