# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `diffusion_lib/` with focused submodules: `trainer/` for training loops, `sampler/` for inference utilities, `method/` for diffusion algorithms, `manager/` for configuration and checkpoint handling, `logger/` for experiment logging, and `evaluator/` for metrics.
- Shared helpers belong in `diffusion_lib/utils/`; prefer mirroring this layout inside `tests/` (e.g., `tests/trainer/test_base_trainer.py`) to keep imports predictable.
- Configuration files and experiment outputs are expected under the directory provided to `FileManager` via `config['save_dir']`; avoid committing large artifacts.

## Build, Test, and Development Commands
- Create a virtual environment and install in editable mode: `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
- Sync dependencies and dev tooling with `pip install -r requirements.txt` followed by `pip install -e .[dev]` when you need linting and typing extras.
- Run unit checks with `pytest tests/ --maxfail=1 -q`; add `--cov=diffusion_lib` when validating coverage.
- Format and lint before pushing: `black diffusion_lib tests`, `flake8 diffusion_lib tests`, and `mypy diffusion_lib` to keep interfaces type-safe.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, `snake_case` functions, and `PascalCase` classes (e.g., `BaseTrainer`, `FileManager`).
- Type annotations are expected for public APIs; match the existing patterns in `diffusion_lib/trainer/base_trainer.py` and `diffusion_lib/manager/file_manager.py`.
- Keep modules focused; when adding new algorithms, extend `method/` and expose them through `diffusion_lib/__init__.py`.

## Testing Guidelines
- Prefer fast, CPU-friendly tests; mock out GPU-only calls and filesystem writes from `FileManager` when practical.
- Name test files `test_<module>.py` and group fixtures in `conftest.py` per package area.
- Include regression tests for checkpoint resume paths, logging side effects, and sampler determinism (`eta=0` branch) to protect critical flows.

## Commit & Pull Request Guidelines
- Use Conventional Commit prefixes (`feat`, `fix`, `docs`, `chore`, etc.) as seen in the existing history (`feat: logger, tgt_distribution getter and condition getter`).
- Each PR should describe motivation, summarize changes, list verification commands run, and link related issues or experiments.
- Update docs or docstrings when APIs change and attach relevant console output or screenshots for UX-facing changes.
