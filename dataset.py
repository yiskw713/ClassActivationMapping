import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
import scipy.io

from PIL import Image, ImageFilter


class PartAffordanceDataset(Dataset):
    """Part Affordance Dataset"""
    
    def __init__(self, csv_file, config, transform=None):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.config = config
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        label_path = self.df.iloc[idx, 1]
        image = Image.open(image_path)
        label = np.load(label_path)
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample



''' transforms for pre-processing '''

def crop_center_pil_image(pil_img, crop_height, crop_width):
    w, h = pil_img.size
    return pil_img.crop(((w - crop_width) // 2,
                        (h - crop_height) // 2,
                        (w + crop_width) // 2,
                        (h + crop_height) // 2))


class CenterCrop(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def __call__(self, sample):
        if 'label' in sample:
            image, label = sample['image'], sample['label']
            image = crop_center_pil_image(image, self.config.height, self.config.width)
            return {'image': image, 'label': label}
            
        else:
            image = sample['image']
            image = crop_center_pil_image(image, self.config.height, self.config.width)
            return {'image': image}



class ToTensor(object):
    def __call__(self, sample):
        
        if 'label' in sample:
            image, label = sample['image'], sample['label']
            return {'image': transforms.functional.to_tensor(image).float(), 
                    'label': torch.from_numpy(label).float()}
        else:
            image = sample['image']
            return {'image': transforms.functional.to_tensor(image).float()}



class Normalize(object):
    def __init__(self, mean=[0.2191, 0.2349, 0.3598], std=[0.1243, 0.1171, 0.0748]):
        self.mean = mean
        self.std = std


    def __call__(self, sample):

        if 'label' in sample:
            image, label = sample['image'], sample['label']
            image = transforms.functional.normalize(image, self.mean, self.std)
            return {'image': image, 'label': label}
        else:
            image = sample['image']
            image = transforms.functional.normalize(image, self.mean, self.std)
            return {'image': image}



''' 
# if you want to calculate mean and std of each channel of the images,
# try this code:

data = PartAffordanceDataset('image_class_path.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor()
                                ]))

data_laoder = DataLoader(data, batch_size=10, shuffle=False)

mean = 0
std = 0
n = 0

for sample in data_laoder:
    img = sample['image']   
    img = img.view(len(img), 3, -1)
    mean += img.mean(2).sum(0)
    std += img.std(2).sum(0)
    n += len(img)
    
mean /= n
std /= ns

'''



'''
# if you also want to calculate class weight,
# please try this code

dataset = PartAffordanceDataset('train_with_label.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor()
                                ]))

data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

cnt_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

for sample in data_loader:
    img = sample['label'].numpy()
    
    num, cnt = np.unique(img, return_counts=True)
    
    for n, c in zip(num, cnt):
        cnt_dict[n] += c

# cnt_dict
# {0: 1151953630, 1: 14085528, 2: 6604904, 3: 5083312,
#  4: 15579160, 5: 2786632, 6: 3814170, 7: 8105464}

class_num = torch.tensor([1151953630, 14085528, 6604904, 5083312,
                        15579160, 2786632, 3814170, 8105464])
total = class_num.sum().item()
frequency = class_num.float() / total
median = torch.median(frequency)
class_weight = median / frequency

'''