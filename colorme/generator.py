# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:55:31 2020

@author: james

Run instructions:
    
1) from Bane import *


assuming you are lame and you call your model generator then:

2) generator = DoYouNetFeelInCharge(INPUT Dimensions) --> Input Dimensinos: List or tuple of batch size, channel size, input image size
                                                   Can also not pass in anything and the model will default to (1,1,64,64)
   
Again assuming you are boring and call results of DoYouNetFeelInCharge, outputs and inputs as inputs then     
                                    
3) outputs = generator(inputs) --> inputs are of the size you passed into the initializer in 2)

RETURNS:

A tensor of (batch,3,H,W) with values bounded between -1 and 1 (tanh activation!)

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import logging


def tanh_onto_0_to_1(x):
    return torch.mul(0.5, torch.add(x, 1))

def sigmoid_onto_0_to_1(x):
    return x

activations_onto_0_to_1 = {
    nn.Tanh: tanh_onto_0_to_1,
    nn.Sigmoid: sigmoid_onto_0_to_1,
}

class FCNGenerator(nn.Module):
    
    def __init__(self, inputDimensions=(1, 1, 64, 64)):
        #input
        # dimensions are a tuple of (B,Depth,H,W) Note (reminder) --> PyTorch uses channels/depth in the 1 index NOT the 3 index!!!!!!
        super(FCNGenerator, self).__init__()
        if inputDimensions[2] != inputDimensions[3]:
            logging.debug("Yo dumbass, learn to feed in a square image!")
        if inputDimensions[2]%2 != 0:
            logging.debug("Do you understand how this works??? Feed in a square image that has h,w dimensions that are powers of 2")
        numLayers = -1
        currentDim = inputDimensions[3]
        while currentDim != 1:
            numLayers +=1
            currentDim = currentDim/2
        logging.debug(f"num_layers: {numLayers}")
        logging.debug("Photocopier Initialize!: Input dimensions are:")
        self.num_layers = numLayers
        logging.debug(inputDimensions)
        logging.debug("\n")
        possible_channel_values = [64, 128, 256, 512, 512, 512, 512]
        input_channels_order = [inputDimensions[1]] + possible_channel_values[:numLayers]
        #logging.debug(relevantInputChannels)

        output_channels_order = [] #[(howDoUFeel[:numLayers][-1], howDoUFeel[:numLayers][-1])] #[howDoUFeel[:numLayers][-1]] + [2*el for el in howDoUFeel[:numLayers][::-1]] # Front is because of bottleneck layer; 3 is the number of color channels
        reversed_channel_values = possible_channel_values[:numLayers][::-1]

        for i in range(len(reversed_channel_values)):
            if i == 0:
                output_channels_order.append((reversed_channel_values[i],reversed_channel_values[i]))
            else:
                output_channels_order.append((reversed_channel_values[i-1]*2, reversed_channel_values[i]))



        #logging.debug(relevantOutputChannels)
        
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        logging.debug("Time to look under the hood of enigma:")
        for i in range(len(input_channels_order) - 1):
            if i == 0:
                #logging.debug(relevantInputChannels[i])
                #logging.debug(relevantInputChannels[i+1])
                self.encoder_layers.append(FCNEncoderLayer(input_channels_order[i], input_channels_order[i + 1], False))
                #self.christopher.append()
            else:
                self.encoder_layers.append(FCNEncoderLayer(input_channels_order[i], input_channels_order[i + 1]))
           
        #bottlenecklayer
        self.encoder_layers.append(FCNEncoderLayer(input_channels_order[len(input_channels_order) - 1], input_channels_order[len(input_channels_order) - 1], False))
        #self.christopher.append()
        #logging.debug("I have gotten through the encoder...")
        logging.debug("\n")
        logging.debug("Christopher what are your magical decoding connections?")
        for i in range(len(output_channels_order)):
            #logging.debug("I have gotten through the decoder...")
            #if i == 0:
            #    self.christopher.append(YoullJustHaveToImagineTheFire(relevantOutputChannels[i],relevantOutputChannels[i]))
            if i < 3: 
                #logging.debug(relevantOutputChannels[i])
                #logging.debug(relevantOutputChannels[i+1])
                self.decoder_layers.append(FCNDecoderLayer(output_channels_order[i][0], output_channels_order[i][1]))
            else:
                self.decoder_layers.append(FCNDecoderLayer(output_channels_order[i][0], output_channels_order[i][1], False))
        
        self.final_layer = nn.ConvTranspose2d(2 * output_channels_order[-1][1], 3, kernel_size= 4, stride = 2, padding = 1)
        self.final_activation = nn.Tanh()
        self.activation_onto_0_to_1 = activations_onto_0_to_1[
            self.final_activation.__class__]
        
    def forward(self, x):
        # logging.debug("Why go forward when you could go sideways?")
        forConcatenation = list()
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x)
            if (i != len(self.encoder_layers) - 1): #don't want the bottleneck layer
                forConcatenation.append(x)
                #logging.debug(x.size())
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x, forConcatenation[len(forConcatenation) - 1 - i])
        
        # logging.debug("MORTAL COMBAT: Finish Him...")
        # logging.debug("\n")
        x = self.final_activation(self.final_layer(x))
        # logging.debug("Winner!!!: Final dimensions are:")
        # logging.debug(x.size())
        x = self.activation_onto_0_to_1(x)
        return x
        

class FCNEncoderLayer(nn.Module): #encoder
    def __init__(self, in_channels, out_channels, batchNorm = True):
        super(FCNEncoderLayer, self).__init__()

        #logging.debug(lotOfHeat)
        #logging.debug(freshOutOfTheOven)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if not batchNorm:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4,
                                padding = 1, stride = 2), self.activation]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 4,
                                padding = 1, stride = 2), nn.BatchNorm2d(
                out_channels), self.activation]
        logging.debug(layers)
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        #logging.debug(x.shape)
        #logging.debug("Running an iteration")
        #logging.debug(self.layer(x).size())
        return self.layers(x)
        
        
class FCNDecoderLayer(nn.Module):#decoder
    def __init__(self, in_channels, out_channels, dropout = True):
        super(FCNDecoderLayer, self).__init__()

        if dropout:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.Dropout(0.5)]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(out_channels)]
        
        logging.debug(layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, concatenated_features):
        #logging.debug("Going Forward")
        x = self.layers(x)
        #logging.debug("AYAYAYAYAAYA")
        #logging.debug(x.shape)
        #logging.debug(selenaKyle.shape)
        x = torch.cat((x, concatenated_features), 1)
        #logging.debug(x.shape)
        return F.relu(x)


if __name__ == "__main__":
    FCNGenerator()
