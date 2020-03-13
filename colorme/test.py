import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from colorme.dataloader import ImageDataset
from colorme.training import load_config, criterions, SSIM_Loss
from pytorch_msssim import ssim as py_ssim
from SSIM_PIL import compare_ssim as pil_ssim
import numpy as np
from skimage import color



# Note: psnr is undefined when images are identical, so we will put
# a 0 into the output tensor for each pair of images where this occurs.
def psnr(tensor1, tensor2, PIX_MAX, use_gpu):
    mse = torch.mean((tensor1 - tensor2) ** 2, dim=(1,2,3))
    psnr = 20 * torch.log10(PIX_MAX / torch.sqrt(mse))

    zeros = torch.full(mse.shape, 0)
    if use_gpu:
        zeros = zeros.cuda()
    return torch.where(mse == 0, zeros, psnr)


def load_model(model_path, use_gpu):
    # Augh...  what is in this checkpoint object that crashes?
    if not use_gpu:
        checkpoint = torch.load(model_path, torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)

    print("Loaded model from global step: {}".format(checkpoint[
                                                         'global_step']))
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

    if "normalize" not in model_args:
        print("WARNING: normalize not specified, assuming:", model.normalize)
    if "color_space" not in model_args:
        print("WARNING: color_space not specified, assuming: ", model.color_space)
    return model


def tensor_to_pil(rgb0to1, index_in_batch):
    img = rgb0to1.cpu()
    img = img.data[index_in_batch].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img

def tensor_to_lab(rgb0to1):
    # INPUT: B C H W
    img = rgb0to1.cpu()
    img = img.numpy()
    img = img.transpose(0, 3, 2, 1)  # B W H C
    img = img * 255
    img = img.astype(np.uint8)
    img = color.rgb2lab(img)
    return img


def _load(config_path, model_path):
    config = load_config(config_path)

    # TODO these should be sorted in the order they're called
    test_data = config["test_data"]
    random_seed = config.get("random_seed", None)
    batch_size = config.get("batch_size", 1)
    eval_mode = config.get("eval_mode", True)
    use_gpu = config.get("use_gpu", torch.cuda.is_available())

    # 0 is default dataloader value for num_workers
    num_workers = config.get("num_workers", 0)

    if random_seed is not None:
        torch.manual_seed(random_seed)

    model = load_model(model_path, use_gpu)

    test_dataset = ImageDataset(path_file=test_data, random_seed=random_seed,
                                color_space=model.color_space,
                                normalize=model.normalize)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if eval_mode:
        # TODO: This switches mode of dropout and batchnorm, but
        #  we might also want to swap batchnorm for instance norm?
        model = model.eval()

    return model, test_dataloader, test_dataset, use_gpu


def eval_test(config_path, model_path, show_image=False):
    model, test_dataloader, test_dataset, use_gpu = _load(config_path, model_path)

    total_processed = 0
    total_disc_real = 0
    total_disc_fake = 0
    total_ssim = 0
    total_psnr = 0
    total_psnr_count = 0 # fails to calculate if images are identical
    total_g_loss = 0
    total_deltaE76 = 0
    total_deltaE94 = 0
    total_deltaE2000 = 0

    total_real_real = 0
    total_fake_fake = 0

    for batch_index, (X_gray, X_color) in enumerate(test_dataloader):
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

        # print(X_fake.shape)
        X_fake_not_norm = test_dataset.invert_transforms(X_fake.cpu()).cuda()
        X_color_not_norm = test_dataset.invert_transforms(X_color.cpu()).cuda()

        psnr_val = psnr(X_fake_not_norm, X_color_not_norm, 1.0, use_gpu)

        lab_fake = tensor_to_lab(X_fake_not_norm)
        lab_real = tensor_to_lab(X_color_not_norm)

        delta_e_76 = color.deltaE_cie76(lab_fake, lab_real)
        delta_e_94 = color.deltaE_ciede94(lab_fake, lab_real)
        delta_e_2000 = color.deltaE_ciede2000(lab_fake, lab_real)

        # TODO:  delta_e_94 has a (some?) nan pixels.
        #  Need to track down if this is due to floating point or what.
        mean_delta_76 = np.nanmean(delta_e_76, axis=(1, 2))
        mean_delta_94 = np.nanmean(delta_e_94, axis=(1, 2))
        mean_delta_2000 = np.nanmean(delta_e_2000, axis=(1, 2))

        # if np.isnan(mean_delta_94).any():
        #     print("ERROR, COMPUTED NAN FOR BATCH:", batch_index)
        #     img_name = test_dataset.data.iloc[batch_index, 0]
        #     print("Maybe on img_name???", img_name)
        #     return

        # print(lab_fake)

        # print("PSNR Tensor: ", psnr_val)
        # Note, if we go with only the py_ssim version of ssim,
        # we can batch and lose the for loop.
        for i in range(X_fake.shape[0]):
            fake_img = tensor_to_pil(X_fake_not_norm, i)
            real_img = tensor_to_pil(X_color_not_norm, i)

            if show_image:
                fake_img.show()
                real_img.show()
                input()

            # TODO:
            #  oh what a nightmare, py_ssim and pil_ssim don't agree.
            #  I'm taking PIL ssim since I know what inputs it expects.
            py_ssim_val = py_ssim(X_fake_not_norm[i:i+1], X_color_not_norm[i:i+1])
            pil_ssim_val = pil_ssim(fake_img, real_img)

            # print("Py ssim: ", py_ssim_val)
            # print("PIL ssim: ", pil_ssim_val)
            ssim_val = pil_ssim_val
            total_ssim += ssim_val

            # Computes exactly the same thing as the range 0-1 batch version, so psnr looks good.
            # psnr2_val = psnr(X_fake_not_norm[i:i+1] * 255, X_color_not_norm[i:i+1] * 255, 255, use_gpu)
            # print("PSNR_INDIV", psnr2_val)


        total_processed += X_fake.shape[0]
        total_disc_real += disc_real.sum()
        total_disc_fake += disc_fake.sum()
        total_g_loss += g_loss.sum()
        total_psnr_count += (psnr_val != 0).sum()
        total_psnr += psnr_val.sum()
        total_deltaE76 += mean_delta_76.sum()
        total_deltaE94 += mean_delta_94.sum()
        total_deltaE2000 += mean_delta_2000.sum()
        total_real_real += (disc_real > .5).sum()
        total_fake_fake += (disc_fake < .5).sum()

        # print(psnr_val)

        print("Avg Disc Sigmoid on REAL: ", total_disc_real / total_processed)
        print("Avg Disc Sigmoid on FAKE: ", total_disc_fake / total_processed)
        print("Avg PIL SSIM: ", total_ssim / total_processed)
        print("Avg psnr: ", total_psnr / total_psnr_count, "dB")
        print("Avg GLoss: ", total_g_loss / total_processed)
        print("Avg delta E 76: ", total_deltaE76 / total_processed)
        print("Avg delta E 94: ", total_deltaE94 / total_processed)
        print("Avg delta E 2000: ", total_deltaE2000 / total_processed)
        print("Correctly Identified Real: ", total_real_real, "/", total_processed)
        print("Correctly Identified Fake: ", total_fake_fake, "/", total_processed)

        if total_psnr_count != total_processed:
            print("Identical Images: ", total_processed - total_psnr_count)
        # Inception Score NOT DOING
        # Fix SSIM CHECK
        # PSNR CHECK
        # Delta E

        # Disc Real on Real
        # Disc Fake on Real
        # Disc Real on Fake
        # Disc Fake on Fake

        print(total_processed, batch_index, "/", len(test_dataloader))
        # print("---X_FAKE:---")
        # print(X_fake.shape)
        # print("---DISC-FAKE:---")
        # print(disc_fake)
        # print("---DISC-REAL:---")
        # print(len(disc_real))


def show_results(config_path, model_path, image_path=None):
    if not image_path:
        eval_test(config_path, model_path, show_image=True)
        return

    model, test_dataloader, test_dataset, use_gpu = _load(config_path, model_path)
    X_gray, X_color = test_dataset.get_image(image_path)

    X_gray = torch.stack([X_gray], dim=0)
    X_color = torch.stack([X_color], dim=0)

    if use_gpu:
        X_gray = X_gray.cuda()
        X_color = X_color.cuda()

    X_fake, disc_fake = model.forward(X_gray, train='none', skip_generator=False)


    X_fake_not_norm = test_dataset.invert_transforms(X_fake.cpu())
    X_color_not_norm = test_dataset.invert_transforms(X_color.cpu())

    fake_img = tensor_to_pil(X_fake_not_norm, 0)
    real_img = tensor_to_pil(X_color_not_norm, 0)

    fake_img.show()
    real_img.show()
    return
