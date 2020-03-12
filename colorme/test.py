import os
import torch
from torch.utils.data import DataLoader
from colorme.dataloader import ImageDataset
from colorme.training import load_config, criterions, SSIM_Loss


def load_model(model_path, use_gpu):
    checkpoint = torch.load(model_path)

    model = checkpoint['model_type'](checkpoint['model_args'])
    model.load_state_dict(checkpoint['state_dict'])

    model.use_gpu = use_gpu    # TODO This feels not particularly robust
    if use_gpu:
        model = model.cuda()

    # TODO:  checkpoints have a Gstate_dict and a Dstate_dict and a Goptimizer
    #  and a Doptimizer, do I need to deal with this?
    return model


def eval_test(config_path, model_path):
    config = load_config(config_path)

    # TODO these should be sorted in the order they're called
    test_data = config["test_data"]
    random_seed = config.get("random_seed", None)
    image_size = config.get("image_size", None)
    batch_size = config.get("batch_size", 1)
    eval_mode = config.get("eval_mode", True)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())
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

    test_dataset = ImageDataset(path_file=test_data, random_seed=random_seed)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = load_model(model_path, use_gpu)

    if eval_mode:
        # TODO: This switches mode of dropout and batchnorm, but
        #  we might also want to swap batchnorm for instance norm?
        model = model.eval()


    for i, (X_gray, X_color) in enumerate(test_dataloader):
        fake_label = torch.full((X_color.size(0),), model.fake_label)
        real_label = torch.full((X_color.size(0),), model.real_label)
        if use_gpu:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        X_fake, disc_fake = model.forward(X_gray, train='none', skip_generator=False)
        disc_real = model.forward(X_color, train='none', skip_generator=True)

        g_loss = model.Gcriterion(X_fake, X_color)

        # Ehh, Dcriterion isn't super informative
        # d_loss_real = model.Dcriterion(disc_fake, fake_label)
        # d_loss_fake = model.Dcriterion(disc_real, real_label)

        print(X_fake.shape())
        print(disc_fake.shape())
        print(disc_real.shape())

        ssim = SSIM_Loss()

        # TODO: wtf is an ssim?
        ssim.forward(X_fake, X_color)
