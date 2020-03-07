import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from colorme.models import BaselineDCN
from colorme.dataloader import ImageDataset
from colorme.generator import FCNGenerator
import yaml

generator_dict = {
    "FCNGenerator": FCNGenerator,
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
    use_gpu = config.get("use_gpu", torch.cuda.is_available())
    lr = config.get("lr",)
    generator_type = config.get('generator', "FCNGenerator")
    generator = generator_dict[generator_type]
    num_epochs = config["num_epochs"]
    summary_interval = config.get("summary_interval", 10)
    validation_interval = config.get("validation_interval", 100)

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

    train_dataloader = ImageDataset(path_file=train_data,
                                    n_samples=subset_size,
                                    random_seed=random_seed,
                                    transform=train_transform,
                                    )

    val_dataloader = ImageDataset(path_file=train_data,
                                  n_samples=subset_size,
                                  random_seed=random_seed,
                                  transform=val_transform,
                                  )

    model = BaselineDCN(n_epochs=num_epochs, lr=lr, logdir=logdir,
                        use_gpu=use_gpu,
                        summary_interval=summary_interval,
                        validation_interval=validation_interval,
                        generator=generator,
                        generator_kwargs=generator_kwargs,
                        )

    model.fit(train_dataloader, val_dataloader)
