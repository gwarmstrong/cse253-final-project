import os
import torch
from torch.utils.data import DataLoader
from colorme.dataloader import ImageDataset
from colorme.training import load_config, criterions, SSIM_Loss


def load_model(model_path, use_gpu):
    # Augh...  what is in this checkpoint object that crashes?
    if not use_gpu:
        checkpoint = torch.load(model_path, torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)

    model_args = checkpoint['model_args']

    # TODO: This. SUUUUUUUUUUUUUUCKS.
    n_epochs = model_args.pop('n_epochs')
    lr = model_args.pop('lr')
    logdir = model_args.pop('logdir')
    summary_interval = model_args.pop('summary_interval')

    model_args.pop("self", None)  # TODO: WHAT I DONT EVEN
    model_args.pop("__class__", None)  # TODO: ARE YOU SERIOUS RIGHT NOW?
    model_args["use_gpu"] = use_gpu  # TODO: I swear...
    print("-----")
    print(model_args)

    model = checkpoint['model_type'](n_epochs, lr, logdir, summary_interval,
                                     **model_args)
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
    batch_size = config.get("batch_size", 1)
    eval_mode = config.get("eval_mode", True)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())

    # 0 is default dataloader value for num_workers
    num_workers = config.get("num_workers", 0)

    # # TODO may want to figure out a way to make this more general
    # input_dimensions = (1, 1, image_size, image_size)
    # generator_kwargs = {'inputDimensions': input_dimensions}
    # logdir = config.get("logdir", os.path.join(os.curdir, 'logs'))

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

    total_processed = 0
    total_disc_real = 0
    total_disc_fake = 0
    total_ssim_loss = 0
    total_g_loss = 0

    ssim = SSIM_Loss()
    for i, (X_gray, X_color) in enumerate(test_dataloader):
        fake_label = torch.full((X_color.size(0),), model.fake_label)
        real_label = torch.full((X_color.size(0),), model.real_label)
        if use_gpu:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
            X_gray = X_gray.cuda()
            X_color = X_color.cuda()

        X_fake, disc_fake = model.forward(X_gray, train='none', skip_generator=False)
        X_color, disc_real = model.forward(X_color, train='none', skip_generator=True)

        g_loss = model.Gcriterion(X_fake, X_color)
        ssim_loss = ssim.forward(X_fake, X_color)

        # Ehh, Dcriterion isn't super informative
        # d_loss_real = model.Dcriterion(disc_fake, fake_label)
        # d_loss_fake = model.Dcriterion(disc_real, real_label)

        total_processed += X_fake.shape[0]
        total_disc_real += disc_real.sum()
        total_disc_fake += disc_fake.sum()
        total_ssim_loss += ssim_loss
        total_g_loss += g_loss.sum()

        print("Avg Disc Sigmoid on REAL: ", total_disc_real / total_processed)
        print("Avg Disc Sigmoid on FAKE: ", total_disc_fake / total_processed)
        print("Avg SSIM_LOSS: ", total_ssim_loss / total_processed)
        print("Avg GLoss: ", total_g_loss / total_processed)

        print(total_processed, "/", len(test_dataloader))
        # print("---X_FAKE:---")
        # print(X_fake.shape)
        # print("---DISC-FAKE:---")
        # print(disc_fake)
        # print("---DISC-REAL:---")
        # print(len(disc_real))


        # TODO: wtf is an ssim?
        # TODO: Need to check that these inputs are correctly shaped, oriented.
