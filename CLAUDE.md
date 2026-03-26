# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLaMA Factory is an efficient fine-tuning framework for 100+ large language models. It supports various training methods (pre-training, SFT, DPO, PPO, KTO, ORPO), multiple backends (PyTorch, vLLM, SGLang), and provides both CLI and Web UI interfaces.

## Development Commands

### Building and Quality Checks
```bash
make build       # Build package
make style       # Auto-fix code style issues (ruff)
make quality     # Check code quality (ruff check + format check)
make test        # Run pytest test suite
make commit      # Install and run pre-commit hooks
make license     # Check Apache 2.0 license headers
```

### Running Tests
- Full test suite: `pytest -vv --import-mode=importlib tests/ tests_v1/`
- Run specific test: `pytest tests/path/to/test_file.py::test_function`
- Tests require GPU for training validation; file-level tests work without GPU
- Disable wandb during testing: `WANDB_DISABLED=true pytest`

### Entry Points
```bash
llamafactory-cli train config.yaml              # Train model
llamafactory-cli webui                         # Launch Gradio Web UI
llamafactory-cli api                           # Launch OpenAI-style API server
llamafactory-cli chat config.yaml              # Interactive chat interface
llamafactory-cli export config.yaml            # Merge/export model
```

Direct Python scripts:
```bash
python src/train.py    # Training
python src/webui.py    # Web UI
python src/api.py      # API server
```

## Code Architecture

### Dual Architecture System

LLaMA Factory has two parallel architectures controlled by `USE_V1` environment variable:

**v0 (default)**: Traditional layered architecture
- Structure: `api`, `webui` → `chat`, `eval`, `train` → `data`, `model` → `hparams` → `extras`
- Entry points delegate to appropriate modules (e.g., `train.py` → `train.tuner`)

**v1 (experimental)**: Plugin-based architecture
- Set `USE_V1=1` to enable
- Structure: `trainers` → `core` → `accelerator`, `plugins`, `config` → `utils`
- Located in `src/llamafactory/v1/`

### Key Modules (v0 Architecture)

**Training Pipeline** (`src/llamafactory/train/`)
- `tuner.py` - Main training entry point
- Subdirectories by training type: `sft/`, `dpo/`, `ppo/`, `rm/`, `kto/`, `pt/`, `mca/`

**Data Processing** (`src/llamafactory/data/`)
- `loader.py` - Dataset loading logic
- `template.py` - Chat template definitions (add custom templates here)
- `processor/` - Data processors for different formats

**Model Support** (`src/llamafactory/model/`)
- `loader.py` - Model loading utilities
- `model_utils/visual.py` - Vision model utilities
- `model_utils/quantization.py` - Quantization support

**Hyperparameters** (`src/llamafactory/hparams/`)
- Dataclass-based configuration system
- Files: `model_args.py`, `data_args.py`, `training_args.py`, `finetuning_args.py`, etc.

**Web UI** (`src/llamafactory/webui/`)
- `interface.py` - Main Gradio interface
- `components/` - UI components
- `manager.py` - Task and model management

**API** (`src/llamafactory/api/`)
- `app.py` - FastAPI application (OpenAI-compatible)

### Configuration System

Training configurations use YAML files (see `examples/` directory):
- Model settings: `model_name_or_path`, `template`, `trust_remote_code`
- Method: `stage` (sft/dpo/ppo/rm/kto/pt), `finetuning_type` (lora/full/freeze)
- Dataset: `dataset`, `cutoff_len`, `preprocessing_num_workers`
- Training: `per_device_train_batch_size`, `learning_rate`, `num_train_epochs`
- Output: `output_dir`, `logging_steps`, `save_steps`, `report_to`

Override config values via CLI: `llamafactory-cli train config.yaml learning_rate=1e-5 batch_size=2`

### Dataset Configuration

Datasets are registered in `data/dataset_info.json`:
- Supports HuggingFace Hub, ModelScope Hub, local files, S3/GCS cloud storage
- Formats: `alpaca` (instruction/input/output) or `sharegpt` (conversations)
- For custom datasets, add entry to `dataset_info.json` before training
- Columns can be customized via `columns` mapping

## Development Practices

### Code Style
- Follow Google Python Style Guide
- Line length: 119 characters
- Indentation: 4 spaces
- Quote style: Double quotes
- Docstrings: Google-style
- Import organization: Known first-party (`llamafactory`), then third-party
- Use 2 blank lines after imports

### Pre-commit Workflow
Before committing:
```bash
make style    # Auto-fix linting issues
make quality  # Verify code quality
make test     # Run tests
```

### Adding Model Support
1. Add model configuration in `src/llamafactory/extras/constants.py`
2. Create model patch in `src/llamafactory/model/` if needed
3. Add template in `src/llamafactory/data/template.py` if new chat format
4. Register in `src/llamafactory/extras/constants.py`

### Adding Training Methods
Training implementations go in `src/llamafactory/train/<method_name>/`

## Environment Variables

- `USE_MODELSCOPE_HUB=1` - Download models from ModelScope instead of HuggingFace
- `USE_OPENMIND_HUB=1` - Download models from Modelers Hub
- `USE_V1=1` - Enable v1 architecture
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `ASCEND_RT_VISIBLE_DEVICES` - NPU selection (Ascend devices)
- `API_PORT` - API server port (default: 8000)
- `GRADIO_SERVER_NAME` - Web UI server address

## Key Dependencies

Core: `torch>=2.4.0`, `transformers>=4.51.0`, `datasets`, `accelerate`, `peft`, `trl`
Training: `deepspeed` (optional), `bitsandbytes` (QLoRA)
Inference: `vllm`, `sglang` (optional)
UI: `gradio>=4.38.0`

## Testing Notes

- Training configurations require GPU machines
- Use `make test` to validate file-level functionality
- End-to-end training tests typically not run in CI
- Disable wandb: `WANDB_DISABLED=true`

## Important File Locations

- Dataset definitions: `data/dataset_info.json`
- Chat templates: `src/llamafactory/data/template.py`
- Model constants: `src/llamafactory/extras/constants.py`
- Hyperparameter definitions: `src/llamafactory/hparams/`
- Training examples: `examples/`
- Version: `src/llamafactory/extras/env.py` (VERSION variable)

## Common Patterns

### Running a Quick Training
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

### Launching Web UI
```bash
llamafactory-cli webui
```

### Starting API Server with vLLM Backend
```bash
API_PORT=8000 llamafactory-cli api examples/inference/qwen3.yaml infer_backend=vllm
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config.yaml
```

### Distributed Training (Multiple Nodes)
```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train config.yaml
```
