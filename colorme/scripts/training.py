import click
from colorme.training import train_baseline


@click.group()
def colorme():
    pass


@colorme.group()
def train():
    pass


@train.command()
@click.option('--config', type=click.Path(exists=True))
def baseline(config):
    train_baseline(config_path=config)
