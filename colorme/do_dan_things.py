#!/usr/bin/env python
# coding: utf-8

# In[1]:
import shutil
import time

import numpy as np
from torch import optim as optim, nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from torch.optim import Adam


from colorme.discriminator import PatchGANDiscriminator


from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import numpy as np
from PIL import Image
import torch
import pandas as pd


def concat_generators(a, b):
    yield from a
    yield from b


class FakeDanGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # print("BEGIN FAKE GENERATION!")
        # print(x.shape)
        output = torch.cat((x, x, x), 1)
        # print(x.shape)
        return output


class DanDataset(Dataset):
    # Uhhhhh... I mean this is close but its not really batchnorm :D
    means = np.array([103.939, 116.779, 123.68]) / 255.
    grayscale_means = np.array([.5])

    def __init__(self, csv_file, use_generator=False):
        self.data = pd.read_csv(csv_file)
        self.use_generator = use_generator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')

        if self.use_generator:
            img = to_grayscale(img)
            img = np.asarray(img)
            # reduce mean
            img = img / 255.
            img[0] -= self.grayscale_means[0]
            img = np.stack([img], axis=0)
        else:
            img = np.asarray(img)
            # reduce mean
            img = img[:, :, ::-1]
            img = np.transpose(img, (2, 0, 1)) / 255.
            img[0] -= self.means[0]
            img[1] -= self.means[1]
            img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        if self.use_generator:
            img_label = torch.tensor(0.)
        else:
            img_label = torch.tensor(1.)

        return img, img_label


def to_grayscale(pil_img):
    return pil_img.convert('L')


def train(discriminator, generator, optimizer, criterion, train_loader, use_gpu,
          epochs=100, total_iter=0, epoch_start=0):
    for epoch in range(epoch_start, epochs):
        for iter, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()  # Move your inputs onto the gpu
                labels = Y.cuda()  # Move your labels onto the gpu
                criterion.cuda()
            else:
                inputs, labels = (X, Y)  # Unpack variables into inputs and labels

            if generator is not None:
                inputs = generator(inputs)
            outputs = discriminator(inputs)
            v_outputs = outputs.view(-1, 1, outputs.shape[2] * outputs.shape[3])
            v_labels = labels.view(-1, 1)
            loss = criterion(v_outputs, v_labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            total_iter += 1

        torch.cuda.empty_cache()
        discriminator.train()


def debug_dat_model(discriminator, generator, img_loader):
    for iter, (X, Y) in enumerate(img_loader):
        if use_gpu:
            inputs = X.cuda()  # Move your inputs onto the gpu
            criterion.cuda()
        else:
            inputs, labels = (X, Y)  # Unpack variables into inputs and labels

        if generator is not None:
            inputs = generator(inputs)
        outputs = discriminator(inputs)
        print("My Output Is: ", outputs)


if __name__ == "__main__":
    discriminator = PatchGANDiscriminator()
    generator = FakeDanGenerator()

    updateable_params = filter(lambda p: p.requires_grad,
                               concat_generators(discriminator.parameters(),
                                                  generator.parameters()))
    learning_rate = .005
    optimizer = optim.Adam(updateable_params, lr=learning_rate)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
    criterion = BCELoss()

    gan_loader = DataLoader(dataset=DanDataset("dan_images.csv",
                                               use_generator=True),
                            batch_size=2,
                            num_workers=4,
                            shuffle=True)
    train_loader = DataLoader(dataset=DanDataset("dan_images.csv"),
                              batch_size=2,
                              num_workers=4,
                              shuffle=True)

    for i in range(1, 10):
        # Train discriminator on color images
        train(discriminator, None, optimizer, criterion, train_loader, use_gpu,
              epochs=1, total_iter=0, epoch_start=0)
        # Train discriminator on fakes
        train(discriminator, generator, optimizer, criterion, gan_loader, use_gpu,
              epochs=10, total_iter=0, epoch_start=0)

    print("These should all be 1")
    debug_dat_model(discriminator, None, train_loader)
    print("These should all be 0")
    debug_dat_model(discriminator, generator, gan_loader)


