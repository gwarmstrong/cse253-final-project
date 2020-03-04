from unittest import TestCase
from colorme.testing.utils import RandomImage
import numpy.testing as npt
import torch


class TestRandomImage(TestCase):

    def test_zeros(self):
        ri = RandomImage(3, 3, mode='zeros')
        obs = ri()
        exp = torch.zeros(3, 3)
        npt.assert_array_equal(obs, exp)

    def test_ones_like(self):
        like = torch.randn(3, 5)
        ri = RandomImage(like, mode='ones_like')
        obs = ri()
        exp = torch.ones(3, 5)
        npt.assert_array_equal(obs, exp)
