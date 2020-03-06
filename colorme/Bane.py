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

class DoYouNetFeelInCharge(nn.Module):
    
    def __init__(self, inputDimensions = (1,1,64,64)): #input dimensions are a tuple of (B,Depth,H,W) Note (reminder) --> PyTorch uses channels/depth in the 1 index NOT the 3 index!!!!!!
        super(DoYouNetFeelInCharge,self).__init__()
        if inputDimensions[2] != inputDimensions[3]:
            print("Yo dumbass, learn to feed in a square image!")
        if inputDimensions[2]%2 != 0:
            print("Do you understand how this works??? Feed in a square image that has h,w dimensions that are powers of 2")
        numLayers = -2
        currentDim = inputDimensions[1]
        while currentDim != 1:
            numLayers +=1
            currentDim = currentDim/2
        print("Photocopier Initialize!: Input dimensions are:")    
        self.deeperThanYouCouldEverImagine = numLayers
        print(inputDimensions)
        print("\n")
        howDoUFeel = [64, 128, 256, 512, 512, 512, 512]
        relevantInputChannels = [inputDimensions[1]] + howDoUFeel[:numLayers]
        #print(relevantInputChannels)
        
        relevantOutputChannels = [] #[(howDoUFeel[:numLayers][-1], howDoUFeel[:numLayers][-1])] #[howDoUFeel[:numLayers][-1]] + [2*el for el in howDoUFeel[:numLayers][::-1]] # Front is because of bottleneck layer; 3 is the number of color channels
        flippyFlippy = howDoUFeel[:numLayers][::-1]

        for i in range(len(flippyFlippy)):
            if i == 0:
                relevantOutputChannels.append((flippyFlippy[i],flippyFlippy[i]))
            else:
                relevantOutputChannels.append((flippyFlippy[i-1]*2, flippyFlippy[i]))



        #print(relevantOutputChannels)
        
        self.enigma = nn.ModuleList()
        self.christopher = nn.ModuleList()
        
        print("Time to look under the hood of enigma:")
        for i in range(len(relevantInputChannels) - 1):
            if i == 0:
                #print(relevantInputChannels[i])
                #print(relevantInputChannels[i+1])
                self.enigma.append(IvePaidYouASmallFortune(relevantInputChannels[i], relevantInputChannels[i+1], False))
                #self.christopher.append()
            else:
                self.enigma.append(IvePaidYouASmallFortune(relevantInputChannels[i], relevantInputChannels[i+1]))
           
        #bottlenecklayer
        self.enigma.append(IvePaidYouASmallFortune(relevantInputChannels[len(relevantInputChannels) - 1], relevantInputChannels[len(relevantInputChannels) - 1], False))
        #self.christopher.append()
        #print("I have gotten through the encoder...")
        print("\n")
        print("Christopher what are your magical decoding connections?")
        for i in range(len(relevantOutputChannels)):
            #print("I have gotten through the decoder...")
            #if i == 0:
            #    self.christopher.append(YoullJustHaveToImagineTheFire(relevantOutputChannels[i],relevantOutputChannels[i]))
            if i < 3: 
                #print(relevantOutputChannels[i])
                #print(relevantOutputChannels[i+1])
                self.christopher.append(YoullJustHaveToImagineTheFire(relevantOutputChannels[i][0],relevantOutputChannels[i][1]))
            else:
                self.christopher.append(YoullJustHaveToImagineTheFire(relevantOutputChannels[i][0],relevantOutputChannels[i][1], False))
        
        self.finishHim = nn.ConvTranspose2d(2*relevantOutputChannels[-1][1], 3, kernel_size= 4, stride = 2, padding = 1)
        self.activateOrderSixtySix = nn.Tanh()
        
    def forward(self, x):
        # print("Why go forward when you could go sideways?")
        forConcatenation = list()
        for i in range(len(self.enigma)):
            x = self.enigma[i](x)
            if (i != len(self.enigma) - 1): #don't want the bottleneck layer
                forConcatenation.append(x)
                #print(x.size())
        for i in range(len(self.christopher)):
            x = self.christopher[i](x, forConcatenation[len(forConcatenation) - 1 - i])
        
        # print("MORTAL COMBAT: Finish Him...")
        # print("\n")
        x = self.activateOrderSixtySix(self.finishHim(x))
        # print("Winner!!!: Final dimensions are:")
        # print(x.size())
        return x
        

class IvePaidYouASmallFortune(nn.Module): #encoder
    def __init__(self, lotOfHeat, freshOutOfTheOven, batchNorm = True):
        super(IvePaidYouASmallFortune, self).__init__()
        
        likeAnOnion = []
        #print(lotOfHeat)
        #print(freshOutOfTheOven)
        
        if not batchNorm:
            likeAnOnion = [nn.Conv2d(lotOfHeat, freshOutOfTheOven, kernel_size = 4, padding = 1, stride = 2),nn.LeakyReLU(0.2,inplace=True)]
        else:
            likeAnOnion = [nn.Conv2d(lotOfHeat, freshOutOfTheOven, kernel_size = 4, padding = 1, stride = 2),nn.BatchNorm2d(freshOutOfTheOven),nn.LeakyReLU(0.2, inplace=True)]
        print(likeAnOnion)
        self.layer = nn.Sequential(*likeAnOnion)


    def forward(self, x):
        #print(x.shape)
        #print("Running an iteration")
        #print(self.layer(x).size())
        return self.layer(x)
        
        
        

class YoullJustHaveToImagineTheFire(nn.Module):#decoder
    def __init__(self, lotOfHeat, freshOutOfTheOven, droppingOutOfSchool = True):
        super(YoullJustHaveToImagineTheFire, self).__init__()
        
        
        oatmealIsEvil = []
        
        if droppingOutOfSchool:
            oatmealIsEvil = [nn.ConvTranspose2d(lotOfHeat,freshOutOfTheOven, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(freshOutOfTheOven), nn.Dropout(0.5)]
        else:
            oatmealIsEvil = [nn.ConvTranspose2d(lotOfHeat,freshOutOfTheOven, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(freshOutOfTheOven)]
        
        print(oatmealIsEvil)
        self.grapefruitIsEvenWorse  = nn.Sequential(*oatmealIsEvil)

     
    
    def forward(self, x, selenaKyle):
        #print("Going Forward")
        x = self.grapefruitIsEvenWorse(x)
        #print("AYAYAYAYAAYA")
        #print(x.shape)
        #print(selenaKyle.shape)
        x = torch.cat((x,selenaKyle),1)
        #print(x.shape)
        return F.relu(x)
        