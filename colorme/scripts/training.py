import sys
import click
import logging
from colorme.training import train_baseline, train_baseline_gan


log_levels = {
    'info' : logging.INFO,
    'INFO' : logging.INFO,
    'debug' : logging.DEBUG,
    'DEBUG' : logging.DEBUG,
    'WARNING' : logging.WARNING,
    'ERROR' : logging.ERROR,
    'CRITICAL' : logging.CRITICAL,
}

@click.group()
def colorme():
    pass


@colorme.group()
def train():
    pass


@train.command()
@click.option('--config', type=click.Path(exists=True))
@click.option('--log', default=None)
@click.option('--logstream', default=sys.stdout)
def baseline(config, log, logstream):
    level = log_levels.get(log, logging.ERROR)
    logging.basicConfig(stream=logstream, level=level)
    train_baseline(config_path=config)


@train.command()
@click.option('--config', type=click.Path(exists=True))
@click.option('--log', default=None)
@click.option('--logstream', default=sys.stdout)
def baseline_gan(config, log, logstream):
    level = log_levels.get(log, logging.ERROR)
    logging.basicConfig(stream=logstream, level=level)
    train_baseline_gan(config_path=config)
