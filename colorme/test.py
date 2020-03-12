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
    model_args["use_gpu"] = use_gpu  # todo: i swear...
    print("-----")
    print(model_args)

    model = checkpoint['model_type'](n_epochs, lr, logdir, summary_interval,
                                     **model_args)
    model.load_state_dict(checkpoint['state_dict'])

    model.use_gpu = use_gpu    # todo this feels not particularly robust
    if use_gpu:
        model = model.cuda()

    # todo:  checkpoints have a gstate_dict and a dstate_dict and a goptimizer
    #  and a doptimizer, do i need to deal with this?
    return model


def eval_test(config_path, model_path):
    config = load_config(config_path)

    # todo these should be sorted in the order they're called
    test_data = config["test_data"]
    random_seed = config.get("random_seed", none)
    batch_size = config.get("batch_size", 1)
    eval_mode = config.get("eval_mode", true)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())

    # 0 is default dataloader value for num_workers
    num_workers = config.get("num_workers", 0)

    # # todo may want to figure out a way to make this more general
    # input_dimensions = (1, 1, image_size, image_size)
    # generator_kwargs = {'inputdimensions': input_dimensions}
    # logdir = config.get("logdir", os.path.join(os.curdir, 'logs'))

    if random_seed is not none:
        torch.manual_seed(random_seed)

    test_dataset = imagedataset(path_file=test_data, random_seed=random_seed)
    test_dataloader = dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=true,
        num_workers=num_workers,
    )

    model = load_model(model_path, use_gpu)

    if eval_mode:
        # todo: this switches mode of dropout and batchnorm, but
        #  we might also want to swap batchnorm for instance norm?
        model = model.eval()

    total_processed = 0
    total_disc_real = 0
    total_disc_fake = 0
    total_ssim_loss = 0
    total_g_loss = 0

    ssim = ssim_loss()
    for i, (x_gray, x_color) in enumerate(test_dataloader):
        fake_label = torch.full((x_color.size(0),), model.fake_label)
        real_label = torch.full((x_color.size(0),), model.real_label)
        if use_gpu:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
            x_gray = x_gray.cuda()
            x_color = x_color.cuda()

        x_fake, disc_fake = model.forward(x_gray, train='none', skip_generator=false)
        x_color, disc_real = model.forward(x_color, train='none', skip_generator=true)

        g_loss = model.gcriterion(x_fake, x_color)
        ssim_loss = ssim.forward(x_fake, x_color)

        # ehh, dcriterion isn't super informative
        # d_loss_real = model.dcriterion(disc_fake, fake_label)
        # d_loss_fake = model.dcriterion(disc_real, real_label)

        total_processed += x_fake.shape[0]
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
