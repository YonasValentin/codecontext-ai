# CodeContext AI™

Privacy-first AI models for automated code documentation generation. Train and run specialized documentation models locally without external dependencies.

## Features

- **Local Processing**: All inference runs locally with no external API calls
- **Specialized Models**: Fine-tuned models for README, API documentation, and changelog generation
- **Privacy Focused**: No data transmission or telemetry collection
- **Production Ready**: Docker support, comprehensive testing, and CI/CD integration

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Generate README documentation
python -m codecontext_ai.cli generate readme /path/to/project

# Train custom model
make train MODEL=readme

# Evaluate model performance
make evaluate MODEL=models/codecontext-readme-7b.gguf
```

## Architecture

### Models

| Model | Purpose | Base | Training |
|-------|---------|------|----------|
| codecontext-readme-7b | Project documentation | CodeLlama-7B | QLoRA fine-tuning |
| codecontext-api-7b | API documentation | CodeLlama-7B | QLoRA fine-tuning |
| codecontext-changelog-7b | Release notes | CodeLlama-7B | QLoRA fine-tuning |

### Training Pipeline

- **Base Model**: CodeLlama-7B with 4-bit quantization
- **Fine-tuning**: Parameter Efficient Fine-Tuning (PEFT) with LoRA
- **Optimization**: GGUF format for efficient local inference
- **Evaluation**: Multi-metric assessment (BLEU, ROUGE-L, semantic similarity)

## Development

### Setup

```bash
# Development environment
make install-dev

# Prepare training data
make prepare-data

# Run tests
make test

# Type checking and linting
make lint && make typecheck
```

### Training

```bash
# Train specific model
make train MODEL=readme

# Train all models
make train-all

# Convert to GGUF format
make convert-gguf MODEL=models/codecontext-readme-7b
```

### Evaluation

```bash
# Evaluate model performance
make evaluate MODEL=models/codecontext-readme-7b.gguf

# Run comprehensive benchmarks
make benchmark
```

## Configuration

Training configurations are stored in `configs/`:
- `readme.yaml`: README model parameters
- `api.yaml`: API documentation model parameters
- `changelog.yaml`: Changelog model parameters

## Privacy

- No external API calls during training or inference
- All processing occurs locally
- Open source codebase for transparency
- No data collection or telemetry

## Performance

Benchmark results on RTX 4090:
- Inference speed: 30-40 tokens/second
- Memory usage: 4-8GB RAM
- Model size: ~4GB per specialized model

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Run test suite: `make test`
5. Submit pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- Issues: [GitHub Issues](https://github.com/YonasValentin/codecontext-ai/issues)
- Discussions: [GitHub Discussions](https://github.com/YonasValentin/codecontext-ai/discussions)