import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from colorme.models import BaselineDCN, BaselineDCGAN
from colorme.dataloader import ImageDataset
from colorme.generator import FCNGenerator
from colorme.discriminator import PatchGANDiscriminator
from pytorch_msssim import SSIM
import yaml


class SSIM_Loss(SSIM):
    def __init__(self):
        super().__init__(data_range=1.0, size_average=True, channel=3)

    def forward(self, img1, img2):
        return 100*(1 - super(SSIM_Loss, self).forward(img1, img2))

generator_dict = {
    "FCNGenerator": FCNGenerator,
}

discriminator_dict = {
    "PatchGANDiscriminator": PatchGANDiscriminator,
}

criterions = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "SSIM_Loss": SSIM_Loss,
}


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def train_baseline(config_path):
    config = load_config(config_path)
    # TODO these should be sorted in the order they're called
    train_data = config["train_data"]
    val_data = config["val_data"]
    subset_size = config.get("subset_size", None)
    if subset_size == -1:
        subset_size = None
    random_seed = config.get("random_seed", None)
    image_size = config.get("image_size", None)
    batch_size = config.get("batch_size", 1)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())
    lr = config.get("lr",)
    generator_type = config.get('generator', "FCNGenerator")
    generator = generator_dict[generator_type]
    num_epochs = config["num_epochs"]
    summary_interval = config.get("summary_interval", 10)
    validation_interval = config.get("validation_interval", 100)
    generator_criterion = config.get("generator_criterion", None)
    if generator_criterion is not None:
        generator_criterion = criterions[generator_criterion]()
    # 0 is default dataloader value for num_workers
    num_workers = config.get("num_workers", 0)

    # TODO may want to figure out a way to make this more general
    input_dimensions = (1, 1, image_size, image_size)
    generator_kwargs = {'inputDimensions': input_dimensions}
    logdir = config.get("logdir", os.path.join(os.curdir, 'logs'))

    if random_seed is not None:
        torch.manual_seed(random_seed)
    if image_size is not None:
        train_transform = [transforms.RandomCrop(image_size),
                           transforms.RandomHorizontalFlip()]
        val_transform = [transforms.CenterCrop(image_size)]
    else:
        train_transform = [transforms.RandomHorizontalFlip()]
        val_transform = None

    train_dataset = ImageDataset(path_file=train_data,
                                 n_samples=subset_size,
                                 random_seed=random_seed,
                                 transform=train_transform,
                                 )

    val_dataset = ImageDataset(path_file=val_data,
                               n_samples=subset_size,
                               random_seed=random_seed,
                               transform=val_transform,
                               )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = BaselineDCN(n_epochs=num_epochs, lr=lr, logdir=logdir,
                        use_gpu=use_gpu,
                        summary_interval=summary_interval,
                        validation_interval=validation_interval,
                        generator=generator,
                        generator_kwargs=generator_kwargs,
                        criterion=generator_criterion,
                        )

    model.fit(train_dataloader, val_dataloader)


def train_baseline_gan(config_path):
    config = load_config(config_path)
    # TODO these should be sorted in the order they're called
    train_data = config["train_data"]
    val_data = config["val_data"]
    subset_size = config.get("subset_size", None)
    if subset_size == -1:
        subset_size = None
    random_seed = config.get("random_seed", None)
    image_size = config.get("image_size", None)
    batch_size = config.get("batch_size", 1)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())
    lr = config.get("lr",)
    generator_type = config.get('generator', "FCNGenerator")
    generator = generator_dict[generator_type]
    num_epochs = config["num_epochs"]
    summary_interval = config.get("summary_interval", 10)
    validation_interval = config.get("validation_interval", 100)
    generator_criterion = config.get("generator_criterion", None)
    if generator_criterion is not None:
        generator_criterion = criterions[generator_criterion]()
    # 0 is default dataloader value for num_workers
    num_workers = config.get("num_workers", 0)

    # TODO may want to figure out a way to make this more general
    input_dimensions = (1, 1, image_size, image_size)
    generator_kwargs = {'inputDimensions': input_dimensions}
    logdir = config.get("logdir", os.path.join(os.curdir, 'logs'))

    # Discriminator specific args
    discriminator_type = config.get('discriminator', "PatchGANDiscriminator")
    discriminator = discriminator_dict[discriminator_type]
    discriminator_kwargs = dict()
    training_loop = config.get('training_loop', 'auto')
    alpha = config.get('alpha', 0.5)

    if random_seed is not None:
        torch.manual_seed(random_seed)
    if image_size is not None:
        train_transform = [transforms.RandomCrop(image_size),
                           transforms.RandomHorizontalFlip()]
        val_transform = [transforms.CenterCrop(image_size)]
    else:
        train_transform = [transforms.RandomHorizontalFlip()]
        val_transform = None

    train_dataset = ImageDataset(path_file=train_data,
                                 n_samples=subset_size,
                                 random_seed=random_seed,
                                 transform=train_transform,
                                 )

    val_dataset = ImageDataset(path_file=val_data,
                               n_samples=subset_size,
                               random_seed=random_seed,
                               transform=val_transform,
                               )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = BaselineDCGAN(n_epochs=num_epochs, lr=lr, logdir=logdir,
                          use_gpu=use_gpu,
                          summary_interval=summary_interval,
                          validation_interval=validation_interval,
                          generator=generator,
                          generator_kwargs=generator_kwargs,
                          generator_criterion=generator_criterion,
                          discriminator=discriminator,
                          discriminator_kwargs=discriminator_kwargs,
                          training_loop=training_loop,
                          alpha=alpha,
                          )

    model.fit(train_dataloader, val_dataloader)


