# Building torch-hailo docs

Sphinx sources for Getting Started, User Guide, and API Reference live in this folder.

## Prerequisites

- Python 3.10+
- This package importable on `PYTHONPATH` (editable install recommended), so autodoc can load `torch_hailo`
- Torch / torchao available in the same environment (API pages import the package)

From the repository root:

```bash
source /path/to/hailo_ai_sw_suite/hailo_venv/bin/activate
pip install -e .
```

## Install doc dependencies

```bash
pip install -r docsrc/requirements.txt
```

Or from this directory:

```bash
cd docsrc
pip install -r requirements.txt
```

## Build HTML

From the repository root:

```bash
sphinx-build -b html docsrc docsrc/_build/html
```

Or with the Makefile from this directory:

```bash
cd docsrc
make html
```

Output: `docsrc/_build/html/index.html` (or `_build/html/index.html` if you ran `make` inside `docsrc`).

Open it in a browser:

```bash
xdg-open docsrc/_build/html/index.html
```

## Clean

```bash
cd docsrc
make clean
```

`_build/` is gitignored and should not be committed.

## GitHub Pages

On every relevant push to `main`, [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) builds this tree and deploys to:

https://soboroo.github.io/torch_hailo/

You can also run the **docs** workflow manually from the Actions tab (`workflow_dispatch`). Hailo SDK is not required for the docs build; only PyTorch CPU wheels and autodoc imports are used.
