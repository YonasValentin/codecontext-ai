# CodeContext AI - Development Makefile

.PHONY: help install install-dev test lint format clean train evaluate benchmark docker

# Default target
help:
	@echo "CodeContext AI Development Commands"
	@echo "=================================="
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  clean        Clean build artifacts and caches"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  lint         Run linting and formatting checks"
	@echo "  format       Format code with black"
	@echo "  typecheck    Run type checking"
	@echo ""
	@echo "Data & Training:"
	@echo "  prepare-data Collect and prepare training datasets"
	@echo "  train        Train models (specify MODEL=readme/api/changelog)"
	@echo "  evaluate     Evaluate trained models"
	@echo "  benchmark    Run comprehensive benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run in Docker container"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Code Quality
test:
	pytest tests/ -v --cov=codecontext_ai --cov-report=html

lint:
	black --check codecontext_ai/ tests/ scripts/
	flake8 codecontext_ai/ tests/ scripts/
	mypy codecontext_ai/

format:
	black codecontext_ai/ tests/ scripts/
	isort codecontext_ai/ tests/ scripts/

typecheck:
	mypy codecontext_ai/

# Data Preparation
prepare-data:
	python scripts/prepare_dataset.py \
		--output ./data \
		--readme-samples 10000 \
		--api-samples 5000 \
		--changelog-samples 3000 \
		--architecture-samples 8000 \
		--implementation-samples 6000 \
		--component-samples 4000 \
		--best-practices-samples 5000

# Training
train:
	@if [ -z "$(MODEL)" ]; then \
		echo "Please specify MODEL (readme/api/changelog/architecture/implementation/component/best_practices/all)"; \
		echo "Example: make train MODEL=readme"; \
		exit 1; \
	fi
	python train.py --config configs/$(MODEL).yaml

train-all:
	python train.py --config configs/readme.yaml
	python train.py --config configs/api.yaml  
	python train.py --config configs/changelog.yaml
	python train.py --config configs/architecture.yaml
	python train.py --config configs/implementation.yaml
	python train.py --config configs/component.yaml
	python train.py --config configs/best_practices.yaml

# Evaluation
evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "Please specify MODEL path"; \
		echo "Example: make evaluate MODEL=models/codecontext-readme-7b.gguf"; \
		exit 1; \
	fi
	python -m codecontext_ai.evaluation --model $(MODEL) --benchmark all

benchmark:
	python scripts/benchmark_all.py \
		--models-dir ./models \
		--output benchmark_results.json \
		--visualize

# Docker
docker-build:
	docker build -t codecontext-ai:latest .

docker-run:
	docker run -it --gpus all \
		-v $(PWD):/workspace \
		-p 8000:8000 \
		codecontext-ai:latest

# Cleanup
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Model Management
convert-gguf:
	@if [ -z "$(MODEL)" ]; then \
		echo "Please specify MODEL path"; \
		exit 1; \
	fi
	python scripts/convert_to_gguf.py \
		--model $(MODEL) \
		--output $(MODEL).gguf \
		--quantization q4_0

upload-hub:
	@if [ -z "$(MODEL)" ] || [ -z "$(REPO)" ]; then \
		echo "Please specify MODEL and REPO"; \
		echo "Example: make upload-hub MODEL=models/codecontext-readme-7b REPO=codecontext/codecontext-readme-7b"; \
		exit 1; \
	fi
	python scripts/upload_to_hub.py \
		--model $(MODEL) \
		--repo $(REPO) \
		--token $(HUGGINGFACE_TOKEN)

# Development helpers
setup-dev: install-dev prepare-data
	@echo "Development environment ready!"

quick-test:
	pytest tests/test_inference.py -v

profile:
	python -m cProfile -o profile.stats scripts/benchmark_all.py --models-dir ./models
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

# CI/CD helpers
ci-test: lint test
	@echo "All CI checks passed!"

release-check: ci-test benchmark
	@echo "Release validation complete!"