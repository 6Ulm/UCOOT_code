# UCOOT

## Install

We recommend using a dedicated `conda` environment.
Then, install project dependencies locally:

```
conda create -n ucoot python=3.8
conda activate ucoot
pip install -r requirements.txt
```

## Development

Code contributed to this project should be `black`- and `flake8`-compatible.
Compatibility of each commit is assessed using `pre-commit`.

To enable this fonction, please run after a successful install:

```
pre-commit install
```
