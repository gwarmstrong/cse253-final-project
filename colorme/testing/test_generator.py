from unittest import TestCase
from unittest import mock
from colorme.starter import BaselineDCN
from colorme.testing.utils import (RandomImage, NaiveConvGenerator,
                                   Small3x3x2x3Dataset, NaiveMultGenerator,
                                   ColormeTestCase,
                                   )
import numpy.testing as npt
import torch
from torch.utils.data import DataLoader


class TestBaselineDCN_forward(TestCase):
    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_forward(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
        pred = gen(X)
        exp = torch.zeros(5, 32, 32, 3)
        # assertion depends on forward call just giving generator output
        npt.assert_array_equal(pred, exp)

    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_forward_no_grad(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
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
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
        bad_arg = 'bad_arg'
        with self.assertRaisesRegex(ValueError, f"Invalid value '{bad_arg}'"):
            gen(X, train=bad_arg)


class TestBaselineDCN_fit(ColormeTestCase):

    package = 'colorme.testing'

    @mock.patch('colorme.starter.BaselineDCN.set_generator')
    def test_fit(self, mock_set_gen):
        torch.manual_seed(725)
        mock_set_gen.return_value = NaiveMultGenerator(shape=(3, 3, 2, 3))
        dat = Small3x3x2x3Dataset()
        dl = DataLoader(dat, batch_size=3)
        val_dl = DataLoader(dat, batch_size=1, shuffle=True)
        logdir = self.create_data_path('test_logs')
        gen = BaselineDCN(n_epochs=100, lr=0.1, logdir=logdir)
        losses = gen.fit(dl, val_dl)
        # the small demo problem is solvable, assert the network has gotten
        # virtually 0 error
        self.assertAlmostEqual(losses[-1], 0, places=4)
