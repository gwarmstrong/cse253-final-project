import torch
import torch.nn as nn
from colorme.generator import InitializeWeights


# A PatchGAN discriminator loosely based on
# https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Input shape: 64 x 64 x 3
        # Output shape: 1x1x1?

        # Inputs must be size 94 x 94 to make a 1x1 output, so padding by 15
        # works.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=15)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0)

        self.collapse = nn.Conv2d(512, 1, kernel_size=1)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.classifier = nn.Sigmoid()

        self.apply(InitializeWeights)

    def forward(self, x):
        # print("-----------")
        # print("BEGIN FORWARD")
        # print(x.shape)
        x = self.bnd1(self.leaky(self.conv1(x)))
        # print(x.shape)
        x = self.bnd2(self.leaky(self.conv2(x)))
        # print(x.shape)
        x = self.bnd3(self.leaky(self.conv3(x)))
        # print(x.shape)
        x = self.bnd4(self.leaky(self.conv4(x)))
        # print(x.shape)
        x = self.leaky(self.conv5(x))
        # print(x.shape)
        x = self.collapse(x)
        # print(x.shape)
        x = self.classifier(x)
        # Depending on image size... this sigmoid will output the wrong shape.
        # So... average pool or wutev?
        # print("-----------")
        # print(x)
        # print("END FORWARD")
        # print("-----------")

        # print("-----------")
        # Compute arithmetic mean of sigmoids... uhh, this seems wrong
        # print("I am well meaning")
        # print(x.shape)
        # x = torch.mean(x, dim=(2,3), keepdim=True)

        # print(x)
        # Compute to geometric mean instead of the arithmetic mean (bcuz these are probabilities)
        num_patches = x.shape[2] * x.shape[3]
        x = torch.prod(x, 2, keepdim=True)
        x = torch.prod(x, 3, keepdim=True)
        x = torch.pow(x, 1.0 / num_patches)
        # print(x.shape)
        # print(x)
        # print("-----------")
        return x
