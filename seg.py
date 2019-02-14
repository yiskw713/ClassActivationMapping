import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import argparse
import numpy as np
import yaml

from addict import Dict

from dataset import PartAffordanceDataset, ToTensor
from dataset import CenterCrop, Normalize, reverse_normalize
from model.resnet import ResNet50_convcam, ResNet50_linearcam, ResNet152_linearcam2
from model.unet import UNet
from cam import CAM, GradCAM, GradCAMpp
from crf import DenseCRF


# assign the colors to each class
colors = torch.tensor([[0, 0, 0],         # class 0 'background'  black
                       [255, 0, 0],       # class 1 'grasp'       red
                       [255, 255, 0],     # class 2 'cut'         yellow
                       [0, 255, 0],       # class 3 'scoop'       green
                       [0, 255, 255],     # class 4 'contain'     sky blue
                       [0, 0, 255],       # class 5 'pound'       blue
                       [255, 0, 255],     # class 6 'support'     purple
                       [255, 255, 255]    # class 7 'wrap grasp'  white
                       ])


# convert class prediction to the mask
def class_to_mask(cls):

    mask = colors[cls].transpose(1, 2).transpose(1, 3)

    return mask


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


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(
        yaml.safe_load(open(args.config)))

    CONFIG.crop_width = 256
    CONFIG.crop_height = 256

    """ DataLoader """
    test_transform = transforms.Compose([
        CenterCrop(CONFIG),
        ToTensor(CONFIG),
        Normalize()
    ])

    test_data = PartAffordanceDataset(
        CONFIG.test_data, config=CONFIG, transform=test_transform, mode='test')

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

    postprocessor = DenseCRF()

    cnt = 0
    intersection = torch.zeros(8)
    union = torch.zeros(8)
    while True:
        print('\n************ loading image ************\n')
        sample = test_iter.next()
        img = sample['image']
        label = sample['label']
        _, _, H, W = img.shape

        print("object ids {}\n".format(sample['obj_label'].nonzero()))
        print("affordance ids {}\n".format(sample['aff_label'].nonzero()))

        # calculate cams
        cams_obj, cams_aff = wrapped_model(img)

        img = reverse_normalize(img)
        images = []    # save an input image and synthesized images with cams
        images.append(img)

        for key, val in cams_obj.items():
            cam_obj = cams_obj[key].squeeze(0)    # shape => (1, H', W')

        _, h, w = cam_obj.shape
        probmap = torch.zeros(
            (1, CONFIG.aff_classes + 1, h, w))    # including bg
        probmap[0, 0] = 0.7    # bg class id is the last index
        for key, val in cams_aff.items():
            val = val.squeeze(0)
            val = val - val.min()
            val = val / val.max()

            probmap[0, key + 1] = \
                torch.where(val > 0.7, val, torch.tensor(0.0))

        probmap = F.interpolate(probmap, size=(H, W), mode='bilinear')
        probmap = F.softmax(probmap, dim=1)
        # probmap = postprocessor(img.numpy().astype(np.uint8).transpose(0, 2, 3, 1).squeeze(0),
        #                         probmap.squeeze(0).numpy())
        # probmap = torch.from_numpy(probmap)
        _, seg = probmap.max(1)    # shape => (1, H, W)
        pred_mask = class_to_mask(seg)
        true_mask = class_to_mask(label)

        for i in range(8):
            seg_i = (seg == i)
            label_i = (label == i)

            inter = (seg_i.byte() & label_i.byte()).float().sum()
            intersection[i] += inter
            union[i] += (seg_i.float().sum() + label_i.float().sum()) - inter

        if cnt < 30:
            save_image(img, CONFIG.result_path + '/img{}.png'.format(cnt))
            save_image(pred_mask, CONFIG.result_path +
                       '/pred_mask{}.png'.format(cnt))
            save_image(true_mask, CONFIG.result_path +
                       '/true_mask{}.png'.format(cnt))

        cnt += 1

        if cnt == 100:
            break

    iou = intersection / union
    m_iou = iou.mean().item()
    print('class_iou: {}\n'.format(iou))
    print('mean_iou: {}\n'.format(m_iou))


if __name__ == '__main__':
    main()
