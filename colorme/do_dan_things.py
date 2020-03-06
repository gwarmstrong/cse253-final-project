#!/usr/bin/env python
# coding: utf-8

# In[1]:
import shutil
import time

import numpy as np
from torch import optim as optim
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

class DanRealDataset(Dataset):
    means = np.array([103.939, 116.779, 123.68]) / 255.

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')
        img = np.asarray(img)

        # reduce mean
        img = img[:, :, ::-1]
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        img_label = torch.tensor(1.)

        return img, img_label

def train(model, optimizer, criterion, train_loader, use_gpu,
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

            outputs = model(inputs)
            v_outputs = outputs.view(-1, 1, outputs.shape[2] * outputs.shape[3])
            v_labels = labels.view(-1, 1)
            loss = criterion(v_outputs, v_labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            total_iter += 1

        torch.cuda.empty_cache()
        model.train()

if __name__ == "__main__":
    model = PatchGANDiscriminator()
    updateable_params = filter(lambda p: p.requires_grad,
                               model.parameters())
    learning_rate = .005
    optimizer = optim.Adam(updateable_params, lr=learning_rate)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    criterion = BCELoss()
    train_loader = DataLoader(dataset=DanRealDataset("dan_images.csv"),
                              batch_size=2,
                              num_workers=4,
                              shuffle=True)




    train(model, optimizer, criterion, train_loader, use_gpu,
          epochs=100, total_iter=0, epoch_start=0)