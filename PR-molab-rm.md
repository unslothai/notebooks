# Remove the molab notebooks and their generator

## Summary

Deletes the marimo (molab) notebook tree and everything that generated or tested it. The `molab/` directory (149 notebooks), `scripts/molab_*.py`, and the `tests/test_molab_*.py` files are removed, along with `tests/_molab_test_utils.py`.

The molab section is removed from `README.md`. The `molab/**` trigger, the marimo/ruff install step, and the molab test steps are removed from `.github/workflows/notebooks-tests-ci.yml`. The molab generation step is removed from the end of `update_all_notebooks.py`, so regenerating notebooks no longer writes `molab/` or edits the README.

## Dependencies

- Removes the `generator` optional dependency group from `pyproject.toml` (`marimo==0.23.8`, `ruff==0.15.16`).
- `uv.lock` drops marimo, ruff and their 17 other dependencies (anyio, click, docutils, h11, idna, itsdangerous, loro, markdown, msgspec, narwhals, pymdown-extensions, python-multipart, pyyaml, starlette, tomlkit, uvicorn, websockets).

## Testing

Not run. Before merging, run `python -m pytest tests -q` and `python update_all_notebooks.py` and confirm the working tree stays clean.
