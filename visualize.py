import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from addict import Dict
from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''
    
    parser = argparse.ArgumentParser(description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu', help='choose a device you want to use')

    return parser.parse_args()

def reverse_normalize(x, mean=[0.2191, 0.2349, 0.3598], std=[0.1243, 0.1171, 0.0748]):
    x[0, :, :] = x[0, :, :] * std[0] + mean[0]
    x[1, :, :] = x[1, :, :] * std[1] + mean[1]
    x[2, :, :] = x[2, :, :] * std[2] + mean[2]
    return x

def show_img(img):
    img = reverse_normalize(img)
    i = img.numpy()
    plt.imshow(np.transpose(i, (1, 2, 0)))
    plt.show()




def main():
    
    args = get_arguments()
    
    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    """ DataLoader """
    test_data = PartAffordanceDataset(CONFIG.test_data,
                                        config=CONFIG,
                                        transform=transforms.Compose([
                                                    CenterCrop(CONFIG),
                                                    ToTensor(),
                                                    Normalize()
                                    ]))

    test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=1)

    test_iter = iter(test_loader)

    model = models.vgg16_bn(pretrained=False)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=7, bias=True)
    model.load_state_dict(torch.load('./result/best_accuracy_model.prm', map_location=lambda storage, loc: storage))
    model.to(args.device)

    while True:
        sample = test_iter.next()
        image, label = sample['image'], sample['label']


        # show images
        show_img(torchvision.utils.make_grid(image))

        # print labels
        print('True labels')
        print(label)

        with torch.no_grad():
            image = image.to(args.device)
            label = label.to(args.device)

            h = model(image)
            h = torch.sigmoid(h)
            h[h>0.5] = 1
            h[h<=0.5] = 0

            total_num = 7 * len(label)
            acc_num = torch.sum(h == label)

            accuracy = float(acc_num) / total_num

        print('\nPredicted labels')
        print(h)

        print('\naccuracy\t{:.3f}'.format(accuracy))
        
        print('\nIf you want to look at more images, input \"c\"')
        s = input()
        if s == 'c':
            continue
        else:
            break

if __name__ == '__main__':
    main()
