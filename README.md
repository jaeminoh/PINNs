## how to build the docs
run `jb build docs`

## how to publish the docs online
run `ghp-import -n -p -f docs/_build/html`

## for inline math mode
If there are underscores `_`, use {math}\`your_math_symbols\` instead of \$ your_math_synbols \$.

## Package manager
uv
- to install, run `pip install uv`

## Local code installation
run `uv pip install -e .`

## JAX install
run `uv pip install "jax[cuda12]"`
or see [JAX - Installation](https://jax.readthedocs.io/en/latest/installation.html)