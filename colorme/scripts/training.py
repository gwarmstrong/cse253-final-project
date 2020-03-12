import sys
import click
import logging
from colorme.training import train_baseline, train_baseline_gan
from colorme.test import eval_test, show_results

log_levels = {
    'debug' : logging.DEBUG,
    'DEBUG' : logging.DEBUG,
    'info' : logging.INFO,
    'INFO' : logging.INFO,
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


@colorme.group()
def test():
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

@test.command()
@click.option('--config', type=click.Path(exists=True))
@click.option('--model', type=click.Path(exists=True))
@click.option('--log', default=None)
@click.option('--logstream', default=sys.stdout)
def eval_gan(config, model, log, logstream):
    level = log_levels.get(log, logging.ERROR)
    logging.basicConfig(stream=logstream, level=level)
    eval_test(config_path=config, model_path=model)

@test.command()
@click.option('--config', type=click.Path(exists=True))
@click.option('--model', type=click.Path(exists=True))
@click.option('--image', type=click.Path(exists=True))
@click.option('--log', default=None)
@click.option('--logstream', default=sys.stdout)
def show_gan(config, model, image, log, logstream):
    level = log_levels.get(log, logging.ERROR)
    logging.basicConfig(stream=logstream, level=level)
    show_results(config_path=config, model_path=model, image_path=image)
