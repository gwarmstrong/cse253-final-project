from unittest import TestCase
from unittest import mock
from colorme.models import BaselineDCN, default_generator, BaselineDCGAN
from colorme.generator import FCNGenerator
import os
from colorme.testing.utils import (RandomImage, NaiveConvGenerator,
                                   Small3x3x2x3Dataset, NaiveMultGenerator,
                                   ColormeTestCase,
                                   CustomSizeRandomDataset,
                                   )
import numpy.testing as npt
import torch
from torch.utils.data import DataLoader


class TestBaselineDCN_forward(TestCase):
    @mock.patch('colorme.models.BaselineDCN.set_generator')
    def test_forward(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
        pred = gen(X)
        exp = torch.zeros(5, 32, 32, 3)
        # assertion depends on forward call just giving generator output
        npt.assert_array_equal(pred, exp)

    @mock.patch('colorme.models.BaselineDCN.set_generator')
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

    @mock.patch('colorme.models.BaselineDCN.set_generator')
    def test_forward_error(self, mock_set_gen):
        image_generator = RandomImage(5, 32, 32)
        X = image_generator()
        mock_set_gen.return_value = RandomImage(5, 32, 32, 3, mode='zeros')
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
        bad_arg = 'bad_arg'
        with self.assertRaisesRegex(ValueError, f"Invalid value '{bad_arg}'"):
            gen(X, train=bad_arg)


class TestBaslineDCN_init(ColormeTestCase):

    def test_default_generator(self):
        torch.manual_seed(725)
        gen = BaselineDCN(n_epochs=10, lr=0.001, logdir='logs')
        self.assertIsInstance(gen.generator, default_generator)

    # def test_generator_input(self):


class TestBaselineDCN_fit(ColormeTestCase):

    package = 'colorme.testing'

    @mock.patch('colorme.models.BaselineDCN.set_generator')
    def test_fit(self, mock_set_gen):
        torch.manual_seed(725)
        mock_set_gen.return_value = NaiveMultGenerator(shape=(3, 3, 2, 3))
        dat = Small3x3x2x3Dataset()
        dl = DataLoader(dat, batch_size=1, shuffle=True)
        val_dl = DataLoader(dat, batch_size=3, shuffle=False)
        logdir = self.create_data_path('test_logs')
        gen = BaselineDCN(n_epochs=100, lr=0.1, logdir=logdir)
        losses = gen.fit(dl, val_dl)
        # the small demo problem is solvable, assert the network has gotten
        # virtually 0 error
        self.assertAlmostEqual(losses[-1], 0, places=4)

    def test_fit_with_actual_model_default(self):
        torch.manual_seed(725)
        dat = CustomSizeRandomDataset((5, 3, 64, 64))
        dl = DataLoader(dat, batch_size=2, shuffle=True)
        val_dl = DataLoader(dat, batch_size=3, shuffle=False)
        logdir = self.create_data_path('test_logs_actual_model')
        gen = BaselineDCN(n_epochs=3, lr=0.1, logdir=logdir)
        gen.fit(dl, val_dl)
        self.assertTrue(os.path.exists(os.path.join(logdir, 'model_best.pth')))


class TestBaselineDCGAN_init(ColormeTestCase):
    def test_dcgan_init_with_defaults(self):
        torch.manual_seed(725)
        gan = BaselineDCGAN(n_epochs=10, lr=0.001, logdir='logs')
        # self.assertIsInstance(gen.generator, default_generator)


class TestBaselineDCGAN_fit(ColormeTestCase):
    package = 'colorme.testing'

    def test_fit_gan(self):
        torch.manual_seed(725)
        dat = CustomSizeRandomDataset((5, 3, 64, 64))
        dl = DataLoader(dat, batch_size=2, shuffle=True)
        val_dl = DataLoader(dat, batch_size=3, shuffle=False)
        logdir = 'test_logs_gan_model'
        logdir = self.create_data_path(logdir)
        gan = BaselineDCGAN(n_epochs=3, lr=0.1, logdir=logdir)
        gan.fit(dl, val_dl)
        self.assertTrue(os.path.exists(os.path.join(logdir, 'model_best.pth')))
