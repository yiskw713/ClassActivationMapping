import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms

import argparse
import numpy as np
import cv2
import yaml

from addict import Dict

from dataset import PartAffordanceDataset, ToTensor, Resize
from dataset import CenterCrop, Normalize, reverse_normalize
from model.resnet import ResNet50_convcam, ResNet50_linearcam, ResNet152_linearcam2
from model.unet import UNet
from model.deeplabv2_linear import DeepLabV2_linear, DeepLabV2_linear_max
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

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(
        yaml.safe_load(open(args.config)))

    """ DataLoader """
    test_transform = transforms.Compose([
        CenterCrop(CONFIG),
        Resize(CONFIG),
        ToTensor(CONFIG),
        Normalize()
    ])

    test_data = PartAffordanceDataset(
        CONFIG.test_data, config=CONFIG, transform=test_transform)

    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=True, num_workers=1)

    test_iter = iter(test_loader)

    """ Load Model """
    if CONFIG.model == "ResNet50_convcam":
        model = ResNet50_convcam(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "ResNet50_linearcam":
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "ResNet152_linearcam2":
        model = ResNet152_linearcam2(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "UNet":
        model = UNet(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "DeepLabV2_linear":
        model = DeepLabV2_linear(
            CONFIG.obj_classes, CONFIG.aff_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        )
    elif CONFIG.model == "DeepLabV2_linear_max":
        model = DeepLabV2_linear_max(
            CONFIG.obj_classes, CONFIG.aff_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        )
    else:
        print('ResNet50_linearcam will be used.')
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)

    state_dict = torch.load(CONFIG.result_path + '/best_accuracy_model.prm',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()

    target_layer_obj = model.conv_obj
    target_layer_aff = model.conv_aff

    # choose CAM, GradCAM or GradCMApp
    # wrapped_model = CAM(model, target_layer_obj, target_layer_aff)
    wrapped_model = GradCAM(model, target_layer_obj, target_layer_aff)
    # wrapped_model = GradCAMpp(model, target_layer_obj, target_layer_aff)
    # wrapped_model = CAM(model, model.feature, model.feature)
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
        # i = input()
        # if i == 'q':
        #     break
        cnt += 1

        if cnt == 50:
            break


if __name__ == '__main__':
    main()
