import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import skimage
import sys
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter

from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize
from model.resnet import ResNet50_convcam, ResNet50_linearcam



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

def show_img(image):
    img = reverse_normalize(image)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()



class SaveFeatures():
    features = None
    def __init__(self, m):
        # register a hook  to save features
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # save features
        self.features = output.to('cpu').data
                        
    def remove(self):
        self.hook.remove()


def getCAM(features, weight_fc):
    '''
    features: feature map before GAP.  shape => (N, C, H, W)
    weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
    cam: class activation map.  shape=> (N, num_classes, H, W)
    '''

    cam = F.conv2d(features, weight=weight_fc[:, :, None, None])    # shape => (1, C, H, W)
    cam = cam.squeeze()
    return cam



def main():
    
    args = get_arguments()
    
    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    """ DataLoader """
    test_data = PartAffordanceDataset(CONFIG.test_data,
                                        config=CONFIG,
                                        transform=transforms.Compose([
                                                    CenterCrop(CONFIG),
                                                    ToTensor(CONFIG),
                                                    Normalize()
                                    ]))

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)
    test_iter = iter(test_loader)

    if CONFIG.model == "ResNet50_convcam":
        model = ResNet50_convcam(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "ResNet50_linearcam":
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)
        final_conv = model._modules.get('layer4')
        activated_features = SaveFeatures(final_conv)
    else:
        print('ResNet50_linearcam will be used.')
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)
        final_conv = model._modules.get('layer4')
        activated_features = SaveFeatures(final_conv)

    model.load_state_dict(torch.load(CONFIG.result_path + '/best_accuracy_model.prm', map_location=lambda storage, loc: storage))
    model.to(args.device)

    cnt = 0

    while True:
        sample = test_iter.next()
        x, y_obj, y_aff = sample['image'], sample['obj_label'], sample['aff_label']

        # show images
        show_img(x)
        plt.savefig(CONFIG.result_path + 'image{}.png'.format(cnt))

        print('True Object\t{}\n'.format(y_obj))
        print('True Affordance\t{}\n'.format(y_aff))
        
        with torch.no_grad():
            x = x.to(args.device)
            y_obj = y_obj.to(args.device)
            y_aff = y_aff.to(args.device)

            h = model(x)    # [ (N, C_obj, H/16, W/16), [(N, C_aff, H/16, W/16), ...]

            # object prediction
            h[0][h[0]>0.5] = 1
            h[0][h[0]<=0.5] = 0
            print('Pred Object\t{}\n'.format(h[0]))

            # affordance prediction
            h[1][h[1]>0.5] = 1
            h[1][h[1]<=0.5] = 0
            print('Pred Affordance\t{}\n'.format(h[1]))
        
        _, _, H, W = h[0].shape

        for o in h[0].nonzero():
            if CONFIG.model == "ResNet50_convcam":
                cam = h[3][o].reshape()
                cam -= torch.min(cam)
                cam /= torch.max(cam).reshape(H, W)
                show_img(x.to("cpu"))
                plt.imshow(cam, alpha=0.5, cmap='jet')
            elif CONFIG.model == "ResNet50_linearcam":
                weight_softmax_params = list(model._modules.get('fc').parameters())


if __name__ == '__main__':
    main()
