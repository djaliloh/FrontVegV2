# Contributing to FrontVeg V2

Contributions, bug reports and feature requests are welcome. By contributing, you agree that your contributions are distributed under the [MIT License](LICENSE.txt).

## Report a bug or request a feature

Open an [issue](https://github.com/djaliloh/FrontVegV2/issues) with:

- what you did, what you expected and what happened, with the full error message;
- your OS, GPU, Python version and the output of `pip list`;
- a sample image when possible.

## Development setup

```bash
git clone --recursive https://github.com/<your-account>/FrontVegV2.git
cd FrontVegV2
pip install -e ".[dev]"
pre-commit install
```

Models and checkpoints are only needed to run the real pipeline; the test suite replaces them with lightweight fakes.

## Checks

```bash
ruff check .          # Lint
ruff format .         # Format
pytest                # Tests (no GPU, no checkpoint needed)
mkdocs serve          # Preview the documentation at http://127.0.0.1:8000
```

The same checks run on every pull request (`.github/workflows/ci.yml`).

## Project layout

```text
src/frontveg/
├── config.py            # FrontVegConfig: every default parameter
├── resources.py         # Locations of external repositories and checkpoints
├── master_pipeline.py   # MasterPipeline: depth + SAM3 + fusion, single entry point
├── cli.py               # frontveg command (main_frontveg.py, main_sam3.py: partial pipelines)
├── core/                # Thresholding, SAM3 strategies, tiling helpers, fusion
├── models/              # Depth-Anything-V2 and SAM3 wrappers (torch)
└── utils/               # Image loading, visualization, plots
src/napari_frontveg/     # napari widget
tests/                   # pytest suite
docs/                    # Documentation (MkDocs)
```

Conventions:

- Images are RGB `uint8` arrays everywhere; conversion to BGR happens only at the OpenCV and Depth-Anything boundaries.
- New parameters go into `FrontVegConfig`, with their default value and a short comment.
- Library code logs with `logging.getLogger(__name__)`, never `print`.
- Code, comments and documentation are in English; comments are short and explain *why*.

## Pull requests

1. Create a branch from `main`.
2. Keep each commit focused, with a message following [Conventional Commits](https://www.conventionalcommits.org/) (`fix:`, `feat:`, `docs:`, `refactor:`, `test:`...).
3. Add or update tests and documentation.
4. Update `CHANGELOG.md` under *Unreleased*.
5. Open the pull request and describe what changes and why.
