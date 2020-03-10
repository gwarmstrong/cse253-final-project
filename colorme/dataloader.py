import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from skimage import color


class ImageDataset(Dataset):

    def __init__(self, path_file, n_samples=None, random_seed=None, transform=None, color_space=None):
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
        if color_space is None:
            self.color = 'RGB'
        else:
            self.color = color_space
    
    
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
        
        gray_normalize = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.458971], std=[0.246539])])
        gray_tensor = gray_normalize(img.copy())
   
        if self.color == 'LAB':
            img = color.rgb2lab(img).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img)
            
        elif self.color == 'RGB':
            rgb_normalize = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                     std=[0.229, 0.224, 0.225])])
                                                
            img_tensor = rgb_normalize(img)
    
        return gray_tensor, img_tensor

    
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
    rgb_dataset = ImageDataset(train_csv, n_samples=5, random_seed=13,
                           transform=transform, color_space='RGB')
    print(rgb_dataset[0])
    
    lab_dataset = ImageDataset(train_csv, n_samples=5, random_seed=13,
                           transform=transform, color_space='LAB')
    print(lab_dataset[0])
