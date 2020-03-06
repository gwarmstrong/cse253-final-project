from unittest import TestCase
from colorme.testing.utils import (RandomImage, Small3x3x2x3Dataset,
                                   NaiveConvGenerator, NaiveMultGenerator,
                                   CustomSizeRandomDataset,
                                   )
import numpy.testing as npt
from torch.utils.data import DataLoader
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


class TestCustomSizeRandomDataset(TestCase):

    def test_methods(self):
        dataset = CustomSizeRandomDataset((3, 2, 20, 4))
        self.assertTupleEqual(dataset[0][0].shape, (1, 20, 4))
        self.assertTupleEqual(dataset[0][1].shape, (2, 20, 4))
        self.assertEqual(len(dataset), 3)


class TestSmall3x3x2x3Dataset(TestCase):

    def test_get_item(self):
        dataset = Small3x3x2x3Dataset()

        sum0, item0 = dataset[0]
        exp0 = torch.zeros(3, 2, 3)
        exp0[0, :, 0] = 1
        npt.assert_array_equal(item0, exp0)
        npt.assert_array_equal(sum0, torch.sum(exp0, 0, keepdim=True))

        sum1, item1 = dataset[1]
        exp1 = torch.zeros(3, 2, 3)
        exp1[1, :, 1] = 1
        npt.assert_array_equal(item1, exp1)
        npt.assert_array_equal(sum1, torch.sum(exp1, 0, keepdim=True))

        sum2, item2 = dataset[2]
        exp2 = torch.zeros(3, 2, 3)
        exp2[2, :, 2] = 1
        npt.assert_array_equal(item2, exp2)
        npt.assert_array_equal(sum2, torch.sum(exp2, 0, keepdim=True))

    def test_dataloader_on_set(self):
        dataset = Small3x3x2x3Dataset()
        dl = DataLoader(dataset, batch_size=1, shuffle=False)
        exp0 = torch.zeros(1, 3, 2, 3)
        exp0[:, 0, :, 0] = 1
        exp1 = torch.zeros(1, 3, 2, 3)
        exp1[:, 1, :, 1] = 1
        exp2 = torch.zeros(1, 3, 2, 3)
        exp2[:, 2, :, 2] = 1
        exps = [exp0, exp1, exp2]
        for i, (gray, color) in enumerate(dl):
            # this_exp = torch.unsqueeze(exps[i], 0)
            this_exp = exps[i]
            # image has the expected RGB
            npt.assert_array_equal(this_exp, color)
            # each image has the correct shape
            self.assertTupleEqual((1, 1, 2, 3), gray.shape)
        # dataloader gives 3 batches
        self.assertEqual(i, 2)


class TestNaiveConvGenerator(TestCase):
    def test_forward(self):
        gen = NaiveConvGenerator()
        dataset = Small3x3x2x3Dataset()
        dl = DataLoader(dataset, batch_size=3, shuffle=False)
        for gray, color in dl:
            obs = gen(gray)
            self.assertTupleEqual(obs.shape, color.shape)


class TestNaiveMultGenerator(TestCase):
    def test_forward(self):
        dataset = Small3x3x2x3Dataset()
        gen = NaiveMultGenerator(shape=dataset[0][0].shape)
        dl = DataLoader(dataset, batch_size=3, shuffle=False)
        for gray, color in dl:
            obs = gen(gray)
            self.assertTupleEqual(obs.shape, color.shape)
