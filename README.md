# cse253-final-project

## Installation

Substituting your GH `username` below, you can clone this repo with
```bash
git clone https://username@github.com/gwarmstrong/cse253-final-project.git
```

Then, install the package with
```bash
cd cse253-final-project
pip install -e .
```

## Training models with colorme
If you want to train a model with `colorme`, you can use the CLI. For example, from the root `colorme` directory,
you can train using the test config in the following way:
```bash
colorme train baseline --config colorme/testing/data/test_generator_config.yml 
```

If you have already installed `colorme`, you may need to re-install it in order for the CLI to work.

## Requirements (so far)

```text
torch>=1.4
numpy
pillow
pandas
click
```

See the [torch page](https://pytorch.org/get-started/locally/) on installation.