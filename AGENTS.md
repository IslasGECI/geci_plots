# AGENTS.md

Working notes for agents in this repository.

## Public interface

The end-user surface is the `geci-plot-cli` command set: each command name, its options, and the output file it writes. The Python modules under `geci_plots/` are internal; a function signature is not part of the interface even though `__init__.py` re-exports it. `CHANGELOG.md` records changes to commands only.

## Commands

- `make install` (`pip install --editable .`) is a prerequisite for every check. Before it runs, `pytest` cannot import `geci_plots`, because rootdir insertion only exposes `tests/`.
- `make tests` runs `pytest --mpl --verbose tests/`. It needs the baselines, which are not tracked, so it fails on a fresh clone until `make setup` runs.
- `make setup` is the only supported way to write baselines: it runs `make clean` (deleting `tests/baseline/`), reinstalls, then regenerates every image from the current code via `pytest --mpl-generate-path tests/baseline/`. It accepts whatever the code draws, so check the figure is intended before committing baselines.
- `make check` chains `make install`, then `black --check` and `flake8` and `mypy` over both `geci_plots/` and `tests/`. `make format` applies `black` only, to the same two trees.
- `make mutants` runs `mutmut` over `geci_plots/` against `tests/` and fails unless `mutmut results` is empty; it depends on `setup`, so it regenerates baselines first.
- `make coverage` is `setup` plus `pytest --cov=geci_plots`.
- `make red`, `make green`, `make refactor` are TDD helpers that commit or `git restore` sources and tests on their own, and run bare `pytest` (no `--mpl`), so they never catch image regressions.

## Testing quirks

- Image tests use `@pytest.mark.mpl_image_compare(tolerance=0, savefig_kwargs={"dpi": 300})`: any change to colour, label, font, or layout fails until baselines are regenerated. Cosmetic edits are behaviour changes here.
- `tests/test_geci_plots.py` also asserts on artists directly (rectangle count, tick count/rotation/size, legend text and font size, axis limits and labels), so renaming a legend label or a column breaks tests.
- Module-level palettes and markers (`islet_colors`, `islet_markers`, `zones_colors`) are pinned by `test_islet_colors` and `test_islet_markers`.
- Fixtures are the CSVs in `tests/data/`; some tests write PNGs into `tests/` and leave them behind. Neither those PNGs nor `tests/baseline/` are tracked.

## Conventions

- GECI style guide (https://islas.dev/guia_de_estilo/STYLEGUIDE): identifiers in English, comments in Spanish, 100-column lines.
- `.flake8` ignores `E203`, `E266`, `E501` and sets `max-complexity = 18`.
- `pyproject.toml` uses the flit backend with a legacy `[tool.flit.metadata]` table and no `[project]` table; commands are declared in `[tool.flit.scripts]`.
- A CLI command is a `@cli.command()` function in `geci_plots/cli.py` whose snake_case name Typer exposes with hyphens (`render_species_...` becomes `render-species-...`). Every command except `version` ends with `plt.savefig(..., transparent=True)` then `plt.close()`.
- `geci_plots/__init__.py` holds `__version__`, the only place the version appears, and re-exports with `from .geci_plots import *`; any import error in the package therefore breaks every test at collection.
- Release commits bump `__version__` and add a dated `## [x.y.z]` section to `CHANGELOG.md` (`c9a1878`, `64e5e6a`); the compare-link block at the bottom is maintained by hand and has drifted, so `7edb488` had to repair links that pointed at another repository.

## Toolchain

- `Dockerfile` installs test tooling only; it never installs the package, so `docker run islasgeci/geci_plots:latest geci-plot-cli ...` fails until `make install` runs inside the container. CI only ever calls make targets that install first.
- CI (`.github/workflows/actions.yml`) builds `islasgeci/geci_plots:{latest,<short sha>}`, runs `make check`, `make coverage` and `make mutants`, force-updates the `latest` git tag, and publishes to PyPI through the `pypi_token` environment using OIDC trusted publishing, so no PyPI token secret exists.
- Base image and pip requirements are unpinned (`FROM python:3`), so dependency drift can change rendered baselines.
- `Dockerfile.dev` builds the `opencode` service and installs `@opencode/cli`; the `islasgeci` service mounts the working copy at `/workdir`.
