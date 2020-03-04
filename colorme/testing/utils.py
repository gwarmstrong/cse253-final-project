import torch


def _positive_rand(*args):
    return torch.abs(torch.randn(*args))


_rand_modes = {'rand': torch.randn, 'ones': torch.ones, 'zeros': torch.zeros,
               'positive_rand': _positive_rand, 'rand_like': torch.randn_like,
               'zeros_like': torch.zeros_like, 'ones_like': torch.ones_like,
               }


class RandomImage:
    def __init__(self, *args, mode='rand'):
        if mode not in _rand_modes:
            raise ValueError(f"Invalid mode '{mode}' for random_image")
        self.args = args
        self.sampler = _rand_modes[mode]

    def __call__(self, *args, **kwargs):
        return self.sampler(*self.args)

