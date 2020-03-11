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

    def __init__(self, path_file, n_samples=None, random_seed=None, transform=None, color_space=None, normalize=True):
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
        transform (list), optional
            Optional transform to be applied on a sample, should be a list of torchvision transforms.
            If None, no transforms to images are applied
        color_space (str), optional
            color space to use for this dataset
            If None, RGB is used
        normalize (bool), optional
            Flag indicating whether to normalize (z-score) or not
            If None, normalize by default
        
        Examples
        --------
        train_csv = 'train_Tiny_ImageNet.csv'
        transform = [transforms.RandomCrop(64),
             transforms.RandomHorizontalFlip()]
        >>> dataset = ImageDataset(train_csv, 
                               n_samples=12, 
                               random_seed=13, 
                               transform=transform, 
                               color_space='RGB', normalize=True)
                               
            
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
        self.normal = normalize
    
    
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
        
        grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor()])        
        gray_tensor = grayscale(img.copy()) 
       
        if self.normal:
            gray_norm = transforms.Normalize(mean=[0.458971], std=[0.246539])  
            gray_tensor = gray_norm(gray_tensor)
        
        if self.color == 'LAB':
            img = color.rgb2lab(img).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img)
            if self.normal:
                lab_normalize = transforms.Normalize(mean=[47.8360,  2.6550,  8.9181], std=[26.8397, 13.3277, 18.7259])
                img_tensor = lab_normalize(img_tensor)
            
        elif self.color == 'RGB':
            tensor = transforms.ToTensor()
            img_tensor = tensor(img)          
            if self.normal:
                rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img_tensor = rgb_normalize(img_tensor)
    
        return gray_tensor, img_tensor

    
    def __len__(self):
        """
        
        Returns
        ------
        length (int) of dataset
        
        """
        
        return len(self.data)

    
def unnormalize_batch(image_batch, color_space='RGB'):
    """     
    
        Parameters
        ----------
        image_batch : (tensor of size (B, C, H, W))
            a batch of normalized images in color_space
        color_space : (str), optional
            the color space to unnormalize from
            
        Return
        ------
        inv_batch : (tensor of size (B, C, H, W))
            a batch of unormalized images in RGB color space
        
    """
    inv_batch = torch.empty(image_batch.size())
    for i, image in enumerate(image_batch):
        if color_space == 'RGB':
            inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                                                     std=[1/0.229, 1/0.224, 1/0.225])
            orig_image = inv_normalize(image)
            
        elif color_space == 'LAB':
            inv_normalize = transforms.Normalize(mean=[-47.8360/26.8397, -2.6550/13.3277, -8.9181/18.7259], 
                                                 std=[1/26.8397, 1/13.3277, 1/18.7259])
            orig_image = color.lab2rgb(inv_normalize(image).numpy().transpose(1, 2, 0))
            orig_image = torch.from_numpy(orig_image.transpose(2, 0, 1))
            
        elif color_space == 'GRAY':
            inv_normalize = transforms.Normalize(mean=[-0.458971/0.246539], 
                                                 std=[1/0.246539])
            orig_image = inv_normalize(image)          
        inv_batch[i] = orig_image
    return inv_batch


def batch_lab2rgb(lab_batch):
    """        
    
        Parameters
        ----------
        lab_batch : (tensor of size (B, C, H, W))
            a batch of LAB images not normalized
            
        Return
        ------
        inv_batch : (tensor of size (B, C, H, W))
            a batch of RGB images not normalized
        
    """
    rgb_batch = torch.empty(lab_batch.size())
    for i, image in enumerate(lab_batch):
        image = color.lab2rgb(image.numpy().transpose(1, 2, 0))
        image = torch.from_numpy(image.transpose(2, 0, 1))
        rgb_batch[i] = image
    return rgb_batch

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
