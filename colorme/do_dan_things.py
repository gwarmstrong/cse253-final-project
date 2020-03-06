#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
print(sys.path)
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
from colorme.Bane import DoYouNetFeelInCharge as Generator


from colorme.discriminator import PatchGANDiscriminator


from torch.utils.data import Dataset, DataLoader  # For custom data-sets
import numpy as np
from PIL import Image
import torch
import pandas as pd

use_gpu = torch.cuda.is_available()

IMG_MEANS = np.array([103.939, 116.779, 123.68]) / 255.
GRAYSCALE_MEANS = np.array([.5])


def cuda_enumerate(loader):
    for iter, (X, Y) in enumerate(loader):
        if use_gpu:
            X = X.cuda()
            Y = Y.cuda()
        yield iter, (X, Y)


def load_loaders():
    train_loader = DataLoader(dataset=DanDataset("dan_images.csv"),
                              batch_size=1,
                              num_workers=4,
                              shuffle=False)
    gan_loader = DataLoader(dataset=DanDataset("dan_images.csv",
                                               use_generator=True),
                              batch_size=1,
                              num_workers=4,
                              shuffle=False)

    return train_loader, gan_loader


def concat_generators(a, b):
    yield from a
    yield from b


def recreate_img(X, grayscale=False):

    X = X.cpu().data.numpy()
    if grayscale:
        X[0] += GRAYSCALE_MEANS[0].item()
        X = np.transpose(X, (0, 2, 3, 1))
    else:
        X[:, 2] += IMG_MEANS[2].item()
        X[:, 1] += IMG_MEANS[1].item()
        X[:, 0] += IMG_MEANS[0].item()
        X = np.transpose(X, (0, 2, 3, 1))
        X = X[:, :, :, ::-1]  # switch back to RGB

    X = X[0]
    if grayscale:
        X = np.concatenate([X,X,X], axis=2)
    X = X * 255.
    X = X.astype(np.uint8)
    return Image.fromarray(X)

class FakeDanGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = torch.cat((x, x, x), 1)
        return output


class DanDataset(Dataset):
    # Uhhhhh... I mean this is close but its not really batchnorm :D

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
            img -= GRAYSCALE_MEANS[0]
            img = np.stack([img], axis=0)
        else:
            img = np.asarray(img)
            # reduce mean
            img = img[:, :, ::-1]
            img = np.transpose(img, (2, 0, 1)) / 255.
            img[0] -= IMG_MEANS[0]
            img[1] -= IMG_MEANS[1]
            img[2] -= IMG_MEANS[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        if self.use_generator:
            img_label = torch.tensor(0.)
        else:
            img_label = torch.tensor(1.)

        return img, img_label


def to_grayscale(pil_img):
    gray = pil_img.convert('L')
    return gray


def train(discriminator, generator, optimizer, criterion, train_loader, use_gpu,
          epochs=100, total_iter=0, epoch_start=0):
    for epoch in range(epoch_start, epochs):
        for iter, (X, Y) in cuda_enumerate(train_loader):
            # Zero EVERYTHING
            discriminator.zero_grad()
            if generator is not None:
                generator.zero_grad()
            optimizer.zero_grad()

            inputs, labels = (X, Y)  # Unpack variables into inputs and labels

            if generator is not None:
                inputs = generator(inputs)
            outputs = discriminator(inputs)
            v_outputs = outputs.view(-1, 1, outputs.shape[2] * outputs.shape[3])
            v_labels = labels.view(-1, 1)
            loss = criterion(v_outputs, v_labels)
            loss.backward()
            # Update the weights for only the subset we're training dependent
            # on which optimizer we pass in.
            optimizer.step()
            torch.cuda.empty_cache()
            total_iter += 1

        torch.cuda.empty_cache()
        discriminator.train()


def debug_dat_model(discriminator, generator, img_loader):
    for iter, (X, Y) in cuda_enumerate(img_loader):
        if generator is not None:
            X = generator(X)
        outputs = discriminator(X)
        print("My Output Is: ", outputs)


def main_train():
    discriminator = PatchGANDiscriminator()
    generator = Generator()

    learning_rate = .005
    updateable_params = filter(lambda p: p.requires_grad,
                               discriminator.parameters())
    discriminator_optimizer = optim.Adam(updateable_params, lr=learning_rate)
    updateable_params = filter(lambda p: p.requires_grad,
                               generator.parameters())
    generator_optimizer = optim.Adam(updateable_params, lr=learning_rate)
    criterion = BCELoss().cuda()

    if use_gpu:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        criterion = criterion.cuda()

    train_loader, gan_loader = load_loaders()

    for i in range(1, 10):
        print(f"GAN EPOCH: {i}")
        # Train discriminator on color images
        train(discriminator, None, discriminator_optimizer, criterion,
              train_loader, use_gpu, epochs=1, total_iter=0, epoch_start=0)
        # Train discriminator on fakes
        train(discriminator, generator, generator_optimizer, criterion,
              gan_loader, use_gpu, epochs=10, total_iter=0, epoch_start=0)
        real_score, fake_score = score_it(discriminator, generator)
        print("REAL SCORE:", real_score)
        print("FAKE SCORE:", fake_score)

    print("These should all be 1")
    debug_dat_model(discriminator, None, train_loader)
    print("These should all be 0")
    debug_dat_model(discriminator, generator, gan_loader)

    now_time = time.strftime("%H-%M-%S")
    os.makedirs("./pickles", exist_ok=True)

    disc_file = f"./pickles/discriminator-{now_time}.p"
    gen_file = f"./pickles/generator-{now_time}.p"
    print("Saving to: ")
    print(disc_file)
    print(gen_file)
    torch.save(discriminator, disc_file)
    torch.save(generator, gen_file)


def score_it(discriminator, generator):
    train_loader, gan_loader = load_loaders()

    rgb_images = enumerate(train_loader)
    gray_images = enumerate(gan_loader)

    real = []
    fake = []
    for checkit in range(5):
        iter, (rgbX, __) = next(rgb_images)
        iter, (grayX, __) = next(gray_images)

        generatedX = generator(grayX)
        rawY = discriminator(rgbX).item()
        generatedY = discriminator(generatedX).item()

        real.append(rawY)
        fake.append(generatedY)

    return np.mean(np.array(real)), np.mean(np.array(fake))

def main_show(discriminator, generator):
    train_loader, gan_loader = load_loaders()

    rgb_images = enumerate(train_loader)
    gray_images = enumerate(gan_loader)
    for checkit in range(5):
        iter, (rgbX, __) = next(rgb_images)
        iter, (grayX, __) = next(gray_images)

        generatedX = generator(grayX)
        rawY = discriminator(rgbX).item()
        generatedY = discriminator(generatedX).item()

        print("RAW")
        recreate_img(rgbX).show()
        print("Real Is Real: ", rawY)
        print("GRAY")
        recreate_img(grayX, grayscale=True).show()
        print("GAN")
        print("Fake is Real: ", generatedY)
        recreate_img(generatedX).show()
        input()



if __name__ == "__main__":
    main_train()
    # main_show(discriminator=torch.load("./pickles/discriminator-20-51-18.p"),
    #           generator=torch.load("./pickles/generator-20-51-18.p"))
