import torch.nn as nn
from utils import init_weights
from model_registry import register_model


# A PatchGAN discriminator loosely based on
# https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
@register_model
class PatchGANDiscriminator(nn.Module):
    def __init__(self, n_class, *args):
        super().__init__()
        self.n_class = n_class

        # Input shape: 64 x 64 x 3
        # Output shape: 1x1x1?

        # Patches are about size 46x46 or so, probably ignores two strips of
        # pixels on the bottom and right.
        self.conv1   = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=0)
        self.bnd5    = nn.BatchNorm2d(512)

        self.leaky = nn.LeakyReLU(inplace=True)
        self.classifier = nn.Sigmoid()

    def forward(self, x):
        print("-----------")
        print("BEGIN FORWARD")
        print(x.shape)
        x = self.bnd1(self.leaky(self.conv1(x)))
        print(x.shape)
        x = self.bnd2(self.leaky(self.conv2(x)))
        print(x.shape)
        x = self.bnd3(self.leaky(self.conv3(x)))
        print(x.shape)
        x = self.bnd4(self.leaky(self.conv4(x)))
        print(x.shape)
        x = self.bnd5(self.leaky(self.conv5(x)))
        print(x.shape)
        x = self.bn1(self.relu(self.deconv1_1(self.deconv1(x))))
        print(x.shape)
        x = self.classifier(x)
        # Depending on image size... this sigmoid will output the wrong shape.
        # So... average pool or wutev?
        print("-----------")
        print(x)
        print("END FORWARD")
        print("-----------")

        return x

    def initialize_weights(self):
        self.apply(init_weights)


