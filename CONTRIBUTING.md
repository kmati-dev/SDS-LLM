# Contributing

## Setup

```bash
pip install -e .[dev]   # install pytest, pytest-cov, pre-commit
pre-commit install      # enable ruff hooks on every commit
```

## Running tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=specdecode --cov-report=term   # with coverage
```

Linting and type checking (ruff / pyright) are coming soon — config is being added in a separate branch.

## Adding a new drafter

```bash
python scripts/new_drafter.py MyDrafter
```

Creates `src/specdecode/simulator/drafter/myDrafter.py` and a matching test file.

## Pull request rules

- All tests must pass (`pytest tests/`)
- CI must be green before merge
- Keep each PR focused — one feature or fix per PR

## Branch protection (repo admin, do once after first CI run on `main`)

GitHub → Settings → Branches → Add rule → Branch name: `main`
→ Enable **Require status checks to pass** → search for `quality` → Save