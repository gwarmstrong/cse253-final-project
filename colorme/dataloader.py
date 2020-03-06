import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

class ImageDataset():
    
    
    def __init__(self, path_file, n_samples=None, random_seed=None, transform=None):
        """
        
        Parameters
        ----------
        path_file : (string)
            file path to a csv file listing all images
        n_samples : (int), optional (default=None)
            number of samples to take out of file path, randomly sampled if specified
            If None, uses all samples (in order from path file)
        random_seed (int), optional
            Used to set random state for reproducable subsampling
            If None, no random state set
        transform (list, optional)
            Optional transform to be applied on a sample, should be a list of torchvision transforms.
            If None, no transforms to images are applied
            
        Examples
        --------
        train_csv = 'train_Tiny_ImageNet.csv'
        transform = [transforms.RandomCrop(64),
             transforms.RandomHorizontalFlip()]
        >>> dataset = ImageDataset(train_csv, n_samples=12, random_seed=13, transform=transform)
            
        """
        if n_samples is None:
            self.data = pd.read_csv(path_file)
        else:
            full_data = pd.read_csv(path_file)
            if random_seed is None:
                self.data = full_data.sample(n_samples)
            else:
                self.data = full_data.sample(n_samples, random_state=random_seed)
        self.transform = transform
    
    
    def __getitem__(self, idx):
        """
        
        Parameters
        ----------
        idx : (int)
            the idx of self.data to grab
            
        Return
        ------
        As tuple:
        grayscale_tensor: 
            torch.Tensor of shape (batch_size, 1, H, W)
        rgb_tensor: 
            torch.Tensor of shape (batch_size, 3, H, W)
        
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data.iloc[idx, 0]        
        im_init = Image.open(img_name).convert('RGB')        
        img = im_init.copy()    
             
        if self.transform is not None:
            transform = transforms.Compose(self.transform)
            img = transform(img)        
        
        grayscale = transforms.Grayscale(num_output_channels=1)
        gray_img = grayscale(img.copy())

        pil2tensor = transforms.ToTensor()
        rgb_tensor = pil2tensor(img)
        gray_tensor = pil2tensor(gray_img)
        
        return gray_tensor, rgb_tensor

    
    def __len__(self):
        """
        
        Returns
        ------
        length (int) of dataset
        
        """
        
        return len(self.data)


if __name__ == "__main__":
    train_csv = 'dan_images.csv'
    transform = [transforms.RandomCrop(64),
                 transforms.RandomHorizontalFlip()]
    dataset = ImageDataset(train_csv, n_samples=5, random_seed=13,
                           transform=transform)
    print(dataset[0])
