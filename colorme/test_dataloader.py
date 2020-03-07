import torch
import argparse
import torchvision.transforms as transforms
import torch.utils.data as data
from colorme.dataloader import ImageDataset
from torch.utils.tensorboard import SummaryWriter


def main():
    writer = SummaryWriter('tensorboard')
    train_csv = 'train_Tiny_ImageNet.csv'
    
    transform = [transforms.RandomCrop(args.crop_size),
             transforms.RandomHorizontalFlip()]
    
    print("Subsetting {0} images from dataset and transforming with random "
          "horizontal flip and random crop to {1}x{1}".format(
                args.num_samples, args.crop_size))
    dataset = ImageDataset(train_csv, n_samples=args.num_samples,
                           random_seed=args.random_seed, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               shuffle=args.shuffle_data)
    
    for i, (gray, rgb) in enumerate(train_loader):
        print('Batch {} Grayscale size:'.format(i+1), gray.size())
        print('Batch {} RGB size:'.format(i+1), rgb.size())
        writer.add_images("Batch {} Grayscale Image".format(i+1), gray, i)
        writer.add_images("Batch {} Color Image".format(i+1), rgb, i)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5, help='Mini-batch size, default is 5')
    parser.add_argument('--shuffle_data', type=bool, default=True, help='Boolean on whether to shuffle data or not, default is true')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of images to subset, default is 10')
    parser.add_argument('--random_seed', type=int, default=13, help='Random seed to test reproducibility, default is 13')
    parser.add_argument('--crop_size', type=int, default=64, help='Random crop size, default is 64x64')
    args = parser.parse_args()
    main()
