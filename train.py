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
import random
import sys
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter

from dataset import PartAffordanceDataset, ToTensor, CenterCrop, Normalize
from model import VGG16


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''
    
    parser = argparse.ArgumentParser(description='adversarial learning for affordance detection')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--device', type=str, default='cpu', help='choose a device you want to use')

    return parser.parse_args()



''' weight initialization '''

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



''' training '''

def full_train(model, sample, criterion, optimizer, device):
    ''' full supervised learning for segmentation network'''
    model.train()

    x, y = sample['image'], sample['label']

    x = x.to(device)
    y = y.to(device)
    
    h = model(x)    # shape => (N, 7)

    loss = criterion(h, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def eval_model(model, test_loader, criterion, config, device):
    ''' calculate the accuracy'''

    model.eval()
    
    total_num = torch.zeros(7).to(device)
    accurate_num = torch.zeros(7).to(device)
    eval_loss = 0.0

    for sample in test_loader:
        x, y = sample['image'], sample['label']

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            h = model(x)     # shape => (N, 7)

            loss = criterion(h, y)
            eval_loss += loss

            h = torch.sigmoid(h)

            h[h>0.5] = 1
            h[h<=0.5] = 0

            total_num += float(len(y))
            accurate_num += torch.sum(h == y, 0).float()


    eval_loss = eval_loss / len(y)

    ''' accuracy of each class'''
    class_accuracy = accurate_num / total_num
    accuracy = torch.sum(accurate_num) / torch.sum(total_num)

    return eval_loss.item(), class_accuracy, accuracy.item()




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
                                                    CenterCrop(CONFIG),
                                                    ToTensor(),
                                                    Normalize()
                                    ]))

    test_data = PartAffordanceDataset(CONFIG.test_data,
                                        config=CONFIG,
                                        transform=transforms.Compose([
                                                    CenterCrop(CONFIG),
                                                    ToTensor(),
                                                    Normalize()
                                    ]))

    train_loader = DataLoader(train_data, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)
    test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)


    model = VGG16(CONFIG.in_channel, CONFIG.out_channel)


    """ optimizer, criterion """
    optimizer = optim.Adam(model.classifier.parameters(), lr=CONFIG.learning_rate)
    
    criterion = nn.BCEWithLogitsLoss()

    losses_train = []
    losses_val = []
    class_accuracy_val = []
    accuracy_val = []
    best_accuracy = 0.0


    for epoch in tqdm.tqdm(range(CONFIG.max_epoch)):

        poly_lr_scheduler(optimizer, CONFIG.learning_rate, 
                        epoch, max_iter=CONFIG.max_epoch, power=CONFIG.poly_power)

        epoch_loss = 0.0

        for sample in train_loader:
            loss_train = full_train(model, sample, criterion, optimizer, args.device)
            
            epoch_loss += loss_train

        losses_train.append(epoch_loss / len(train_loader))

        # validation
        loss_val, class_accuracy, accuracy = eval_model(model, test_loader, criterion, CONFIG, args.device)
        losses_val.append(loss_val)
        class_accuracy_val.append(class_accuracy)
        accuracy_val.append(accuracy)


        if best_accuracy < accuracy_val[-1]:
            best_accuracy = accuracy_val[-1]
            torch.save(model.state_dict(), CONFIG.result_path + '/best_accuracy_model.prm')

        if epoch%50 == 0 and epoch != 0:
            torch.save(model.state_dict(), CONFIG.result_path + '/epoch_{}_model.prm'.format(epoch))

        if writer is not None:
            writer.add_scalars("loss", {'loss_train': losses_train[-1],
                                        'loss_val': losses_val[-1]}, epoch)
            writer.add_scalar("accuracy", accuracy_val[-1], epoch)
            writer.add_scalars("class_accuracy", {
                                                'accuracy of class 1': class_accuracy_val[-1][0],
                                                'accuracy of class 2': class_accuracy_val[-1][1],
                                                'accuracy of class 3': class_accuracy_val[-1][2],
                                                'accuracy of class 4': class_accuracy_val[-1][3],
                                                'accuracy of class 5': class_accuracy_val[-1][4],
                                                'accuracy of class 6': class_accuracy_val[-1][5],
                                                'accuracy of class 7': class_accuracy_val[-1][6],
                                                }, epoch)

        print('epoch: {}\tloss_train: {:.5f}\tloss_val: {:.5f}\taccuracy: {:.5f}'
            .format(epoch, losses_train[-1], losses_val[-1], accuracy_val[-1]))


    torch.save(model.state_dict(), CONFIG.result_path + '/final_model.prm')


if __name__ == '__main__':
    main()