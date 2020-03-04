import torch
from torch import nn
from torch.utils.data import Dataset


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


class Small3x3x2x3Dataset(Dataset):

    def __init__(self):
        # represents 3, 2 x 3 RGB images
        self.data = torch.zeros(3, 3, 2, 3)
        # set first column of first image in red channel to ones
        self.data[0, 0, :, 0] = 1
        self.data[1, 1, :, 1] = 1
        self.data[2, 2, :, 2] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.sum(self.data[item], 0, keepdim=True), self.data[item]


class NaiveConvGenerator(nn.Module):

    def __init__(self, kernel_size=(1, 1), in_channels=1, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              bias=False)

    def forward(self, X):
        return self.conv(X)


class NaiveMultGenerator(nn.Module):

    # in_channels assumed to be 1
    def __init__(self, shape, out_channels=3):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_channels, shape[-1],
                                                shape[-1]),
                                    requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.weights)
