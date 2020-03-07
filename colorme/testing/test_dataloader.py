from colorme.testing.utils import ColormeTestCase
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from colorme.dataloader import ImageDataset
import logging


class TestDataLoader(ColormeTestCase):
    package = 'colorme.testing'

    def setUp(self):
        super().setUp()
        self.crop_size = 32
        self.num_samples = 4
        self.random_seed = 725
        self.batch_size = 3
        self.shuffle_data = False
        train_csv = self.get_data_path('dan_images.csv')

        transform = [transforms.RandomCrop(self.crop_size),
                     transforms.RandomHorizontalFlip()]

        print("Subsetting {0} images from dataset and transforming with "
              "random horizontal flip and random crop to {1}x{1}".format(
                    self.num_samples, self.crop_size))
        import os
        logging.debug(os.path.abspath(os.curdir))
        self.dataset = ImageDataset(train_csv, n_samples=self.num_samples,
                                    random_seed=self.random_seed,
                                    transform=transform)

    def test_dataset_properties(self):
        self.assertEqual(4, len(self.dataset))
        gray2, rgb2 = self.dataset[2]
        self.assertTupleEqual(gray2.shape, (1, 32, 32))
        self.assertTupleEqual(rgb2.shape, (3, 32, 32))

    def test_dataloader_integration(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=0,
                                                   shuffle=self.shuffle_data)

        for i, (gray, rgb) in enumerate(train_loader):
            if i == 0:
                self.assertTupleEqual(gray.size(), (3, 1, 32, 32))
                self.assertTupleEqual(rgb.size(), (3, 3, 32, 32))
            elif i == 1:
                self.assertTupleEqual(gray.size(), (1, 1, 32, 32))
                self.assertTupleEqual(rgb.size(), (1, 3, 32, 32))
