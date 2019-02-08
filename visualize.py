import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid

import argparse
import numpy as np
import cv2
import sys
import yaml

from addict import Dict

from dataset import PartAffordanceDataset, ToTensor
from dataset import CenterCrop, Normalize, reverse_normalize
from model.resnet import ResNet50_convcam, ResNet50_linearcam, ResNet152_linearcam2
from cam import CAM, GradCAM, GradCAMpp


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='choose a device you want to use')

    return parser.parse_args()


def reverse_normalize(x, mean=[0.2191, 0.2349, 0.3598], 
                      std=[0.1243, 0.1171, 0.0748]):
    x[0, :, :] = x[0, :, :] * std[0] + mean[0]
    x[1, :, :] = x[1, :, :] * std[1] + mean[1]
    x[2, :, :] = x[2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear')
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap+img.cpu()
    result = result.div(result.max())

    return result


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(
        yaml.safe_load(open('./result/ResNet50_linearcam/config.yaml')))

    """ DataLoader """
    test_transform = transforms.Compose([
        CenterCrop(CONFIG),
        ToTensor(CONFIG),
        Normalize()
    ])

    test_data = PartAffordanceDataset(
        CONFIG.test_data, config=CONFIG, transform=test_transform)

    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=1)

    test_iter = iter(test_loader)

    """ Load Model """
    model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)
    state_dict = torch.load(CONFIG.result_path + '/best_accuracy_model.prm',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer = model.feature[-1][-1].conv3

    # choose CAM, GradCAM or GradCMApp
    # wrapped_model = CAM(model, target_layer)
    # wrapped_model = GradCAM(model, target_layer)
    wrapped_model = GradCAMpp(model, target_layer)

    cnt = 0
    while True:
        print('\n************ loading image ************\n')
        sample = test_iter.next()
        img = sample['image']
        print("object ids {}\n".format(sample['obj_label'].nonzero()))
        print("affordance ids {}\n".format(sample['aff_label'].nonzero()))

        # calculate cams
        cams_obj, cams_aff = wrapped_model(img)

        img = reverse_normalize(img)
        images = []    # save an input image and synthesized images with cams
        images.append(img)

        for key, val in cams_obj.items():
            result = visualize(img, val)
            images.append(result)

        for key, val in cams_aff.items():
            result = visualize(img, val)
            images.append(result)

        images = make_grid(torch.cat(images, 0))
        save_image(images, CONFIG.result_path + '/result{}.png'.format(cnt))

        print('\nIf you want to quit, please press q. Else, press the others\n')
        i = input()
        if i == 'q':
            break
        cnt += 1


if __name__ == '__main__':
    main()
