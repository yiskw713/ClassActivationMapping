import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import argparse
import numpy as np
import pandas as pd
import random
import sys
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter

from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize
from dataset import Resize, RandomFlip, RandomRotate, ColorChange
from model.resnet import ResNet50_convcam, ResNet50_linearcam, ResNet152_linearcam2


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu', help='choose a device you want to use')

    return parser.parse_args()


''' training '''


def full_train(model, sample, criterion, optimizer, device):
    ''' full supervised learning for segmentation network'''
    model.train()

    x, y_obj, y_aff = sample['image'], sample['obj_label'], sample['aff_label']

    x = x.to(device)
    y_obj = y_obj.to(device)
    y_aff = y_aff.to(device)

    h = model(x)    # h[0] => object, h[1] => affordance

    loss_obj = criterion(h[0], y_obj)
    loss_aff = criterion(h[1], y_aff)
    loss = loss_obj + loss_aff

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_obj.item(), loss_aff.item()


def eval_model(model, test_loader, criterion, config, device):
    ''' calculate the accuracy'''

    model.eval()

    obj_total_num = torch.zeros(config.obj_classes).to(device)
    obj_accurate_num = torch.zeros(config.obj_classes).to(device)
    aff_total_num = torch.zeros(config.aff_classes).to(device)
    aff_accurate_num = torch.zeros(config.aff_classes).to(device)
    loss_obj = 0.0
    loss_aff = 0.0

    for sample in test_loader:
        x, y_obj, y_aff = sample['image'], sample['obj_label'], sample['aff_label']

        x = x.to(device)
        y_obj = y_obj.to(device)
        y_aff = y_aff.to(device)

        with torch.no_grad():
            h = model(x)    # h[0] => object, h[1] => affordance

            loss_obj += criterion(h[0], y_obj)
            loss_aff += criterion(h[1], y_aff)

            h0 = torch.sigmoid(h[0])
            h1 = torch.sigmoid(h[1])

            h0[h0>0.5] = 1
            h0[h0<=0.5] = 0

            h1[h1>0.5] = 1
            h1[h1<=0.5] = 0

            obj_total_num += float(len(y_obj))
            obj_accurate_num += torch.sum(h0 == y_obj, 0).float()

            aff_total_num += float(len(y_aff))
            aff_accurate_num += torch.sum(h1 == y_aff, 0).float()

    loss_obj /= len(test_loader)
    loss_aff /= len(test_loader)

    ''' accuracy of each class'''
    obj_class_accuracy = obj_accurate_num / obj_total_num
    obj_accuracy = torch.sum(obj_accurate_num) / torch.sum(obj_total_num)

    aff_class_accuracy = aff_accurate_num / aff_total_num
    aff_accuracy = torch.sum(aff_accurate_num) / torch.sum(aff_total_num)

    return [loss_obj.item(), obj_class_accuracy, obj_accuracy.item(), 
            loss_aff.item(), aff_class_accuracy, aff_accuracy.item()
            ]


''' learning rate scheduler '''


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """

    if iter % lr_decay_iter or iter > max_iter:
        pass
    else:
        lr = init_lr*(1 - iter/max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():

    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    """ DataLoader """

    train_data = PartAffordanceDataset(CONFIG.train_data,
                                       config=CONFIG,
                                       transform=transforms.Compose([
                                                    RandomRotate(45),
                                                    CenterCrop(CONFIG),
                                                    Resize(CONFIG),
                                                    RandomFlip(),
                                                    ColorChange(),
                                                    ToTensor(CONFIG),
                                                    Normalize()
                                       ]))

    test_data = PartAffordanceDataset(CONFIG.test_data,
                                        config=CONFIG,
                                        transform=transforms.Compose([
                                                    CenterCrop(CONFIG),
                                                    Resize(CONFIG),
                                                    ToTensor(CONFIG),
                                                    Normalize()
                                    ]))

    train_loader = DataLoader(train_data, batch_size=CONFIG.batch_size,
                              shuffle=True, num_workers=CONFIG.num_workers)
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size,
                             shuffle=False, num_workers=CONFIG.num_workers)

    if CONFIG.model == "ResNet50_convcam":
        model = ResNet50_convcam(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "ResNet50_linearcam":
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)
    elif CONFIG.model == "ResNet152_linearcam2":
        model = ResNet152_linearcam2(CONFIG.obj_classes, CONFIG.aff_classes)
    else:
        print('ResNet50_linearcam will be used.')
        model = ResNet50_linearcam(CONFIG.obj_classes, CONFIG.aff_classes)

    model.to(args.device)

    """ optimizer, criterion """

    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    losses_train = []
    losses_train_obj = []
    losses_train_aff = []
    losses_val = []
    losses_val_obj = []
    losses_val_aff = []

    obj_class_accuracy_val = []
    obj_accuracy_val = []
    aff_class_accuracy_val = []
    aff_accuracy_val = []
    best_accuracy = 0.0

    for epoch in tqdm.tqdm(range(CONFIG.max_epoch)):

        poly_lr_scheduler(optimizer, CONFIG.learning_rate, 
                          epoch, max_iter=CONFIG.max_epoch, power=CONFIG.poly_power)

        epoch_loss = 0.0
        epoch_loss_obj = 0.0
        epoch_loss_aff = 0.0

        for sample in train_loader:
            loss_train_obj, loss_train_aff = full_train(model, sample, criterion, optimizer, args.device)

            epoch_loss_obj += loss_train_obj
            epoch_loss_aff += loss_train_aff
            epoch_loss = epoch_loss + loss_train_obj + loss_train_aff

        losses_train_obj.append(epoch_loss_obj / len(train_loader))
        losses_train_aff.append(epoch_loss_aff / len(train_loader))
        losses_train.append(epoch_loss / len(train_loader))

        # validation
        loss_val_obj, obj_class_accuracy, obj_accuracy, loss_val_aff, aff_class_accuracy, aff_accuracy = eval_model(model, test_loader, criterion, CONFIG, args.device)
        losses_val_obj.append(loss_val_obj)
        losses_val_aff.append(loss_val_aff)
        losses_val.append(loss_val_obj + loss_val_aff)
        
        obj_class_accuracy_val.append(obj_class_accuracy)
        obj_accuracy_val.append(obj_accuracy)
        aff_class_accuracy_val.append(aff_class_accuracy)
        aff_accuracy_val.append(aff_accuracy)

        if best_accuracy < (obj_accuracy_val[-1] + aff_accuracy_val[-1]):
            best_accuracy = obj_accuracy_val[-1] + aff_accuracy_val[-1]
            torch.save(model.state_dict(), CONFIG.result_path + '/best_accuracy_model.prm')

        if epoch%50 == 0 and epoch != 0:
            torch.save(model.state_dict(), CONFIG.result_path + '/epoch_{}_model.prm'.format(epoch))

        if writer is not None:
            writer.add_scalars("loss", {'loss_train': losses_train[-1],
                                        'loss_val': losses_val[-1]}, epoch)
            writer.add_scalars("loss", {'loss_train_obj': losses_train_obj[-1],
                                        'loss_val_obj': losses_val_obj[-1]}, epoch)
            writer.add_scalars("loss", {'loss_train_aff': losses_train_aff[-1],
                                        'loss_val_aff': losses_val_aff[-1]}, epoch)
            writer.add_scalars("loss", {'obj_accuracy': obj_accuracy_val[-1],
                                        'aff_accuracy': aff_accuracy_val[-1]}, epoch)
            writer.add_scalars("obj_class_accuracy", {
                                                        'accuracy of class 0': obj_class_accuracy_val[-1][0],
                                                        'accuracy of class 1': obj_class_accuracy_val[-1][1],
                                                        'accuracy of class 2': obj_class_accuracy_val[-1][2],
                                                        'accuracy of class 3': obj_class_accuracy_val[-1][3],
                                                        'accuracy of class 4': obj_class_accuracy_val[-1][4],
                                                        'accuracy of class 5': obj_class_accuracy_val[-1][5],
                                                        'accuracy of class 6': obj_class_accuracy_val[-1][6],
                                                        'accuracy of class 7': obj_class_accuracy_val[-1][7],
                                                        'accuracy of class 8': obj_class_accuracy_val[-1][8],
                                                        'accuracy of class 9': obj_class_accuracy_val[-1][9],
                                                        'accuracy of class 10': obj_class_accuracy_val[-1][10],
                                                        'accuracy of class 11': obj_class_accuracy_val[-1][11],
                                                        'accuracy of class 12': obj_class_accuracy_val[-1][12],
                                                        'accuracy of class 13': obj_class_accuracy_val[-1][13],
                                                        'accuracy of class 14': obj_class_accuracy_val[-1][14],
                                                        'accuracy of class 15': obj_class_accuracy_val[-1][15],
                                                        'accuracy of class 16': obj_class_accuracy_val[-1][16],
                                                        }, epoch)
            # ignore background class
            writer.add_scalars("aff_class_accuracy", {
                                                        'accuracy of class 1': aff_class_accuracy_val[-1][0],
                                                        'accuracy of class 2': aff_class_accuracy_val[-1][1],
                                                        'accuracy of class 3': aff_class_accuracy_val[-1][2],
                                                        'accuracy of class 4': aff_class_accuracy_val[-1][3],
                                                        'accuracy of class 5': aff_class_accuracy_val[-1][4],
                                                        'accuracy of class 6': aff_class_accuracy_val[-1][5],
                                                        'accuracy of class 7': aff_class_accuracy_val[-1][6],
                                                        }, epoch)

        print('epoch: {}\tloss_train: {:.5f}\tloss_val: {:.5f}\tobj_accuracy: {:.5f}\taff_accuracy: {:.5f}'
              .format(epoch, losses_train[-1], losses_val[-1], obj_accuracy_val[-1], aff_accuracy_val[-1]))

    torch.save(model.state_dict(), CONFIG.result_path + '/final_model.prm')


if __name__ == '__main__':
    main()