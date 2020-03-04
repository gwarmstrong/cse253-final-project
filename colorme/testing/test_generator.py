from unittest import TestCase
from unittest import mock
from colorme.starter import BaselineDCN
from colorme.testing.utils import RandomImage
import numpy.testing as npt
import torch


class TestBaselineDCN_forward(TestCase):
    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_forward(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, learning_rate=0.001, logdir='logs')
        pred = gen(X)
        exp = torch.zeros(5, 32, 32, 3)
        # assertion depends on forward call just giving generator output
        npt.assert_array_equal(pred, exp)

    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_forward_no_grad(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, learning_rate=0.001, logdir='logs')
        with mock.patch.object(torch, 'no_grad') as mock_no_grad:
            pred = gen(X, train='none')
            mock_no_grad.assert_called_once()

        exp = torch.zeros(5, 32, 32, 3)
        # assertion depends on forward call just giving generator output
        npt.assert_array_equal(pred, exp)

    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_forward_error(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, learning_rate=0.001, logdir='logs')
        bad_arg = 'bad_arg'
        with self.assertRaisesRegex(ValueError, f"Invalid value '{bad_arg}'"):
            gen(X, train=bad_arg)
