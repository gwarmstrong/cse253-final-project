import torch


def _positive_rand(*args):
    return torch.abs(torch.randn(*args))


_rand_modes = {'rand': torch.randn, 'ones': torch.ones, 'zeros': torch.zeros,
               'positive_rand': _positive_rand}


def random_grayscale_image(n, h, w, mode='rand'):
    if mode not in _rand_modes:
        raise ValueError(f"Invalid mode '{mode}' for random_grayscale_image")
    sampler = _rand_modes[mode]
    return sampler(n, h, w)


def random_color_image(n, h, w, d, mode='rand'):
    if mode not in _rand_modes:
        raise ValueError(f"Invalid mode '{mode}' for random_grayscale_image")
    sampler = _rand_modes[mode]
    return sampler(n, h, w, d)
