import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50_linearcam(nn.Module):

    def __init__(self, obj_classes, aff_classes, pretrained=True):
        super().__init__()

        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature = nn.Sequential(*list(resnet50.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_obj = nn.Linear(2048, obj_classes)
        self.fc_aff = nn.Linear(2048, aff_classes)

        for m in [self.fc_obj, self.fc_aff]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.gap(x)

        x = x.view(x.shape[0], -1)

        y_obj = self.fc_obj(x)
        y_aff = self.fc_aff(x)

        return [y_obj, y_aff]


class ResNet50_convcam(nn.Module):

    def __init__(self, obj_classes, aff_classes, pretrained=True):
        super().__init__()

        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature = nn.Sequential(*list(resnet50.children())[:-2])
        self.conv_obj = nn.Conv2d(2048, obj_classes, (1, 1), stride=1)
        self.conv_aff = nn.Conv2d(2048, aff_classes, (1, 1), stride=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        for m in [self.conv_obj, self.conv_aff]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)

        cam_obj = self.conv_obj(x)
        y_obj = self.gap(cam_obj)
        y_obj = y_obj.view(x.shape[0], -1)

        cam_aff = self.conv_aff(x)
        y_aff = self.gap(cam_aff)
        y_aff = y_aff.view(x.shape[0], -1)

        return [y_obj, y_aff, cam_obj, cam_aff]


class ResNet50_linearcam(nn.Module):

    def __init__(self, obj_classes, aff_classes, pretrained=True):
        super().__init__()

        resnet50 = models.resnet50(pretrained=pretrained)
        self.feature = nn.Sequential(*list(resnet50.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_obj = nn.Linear(2048, obj_classes)
        self.fc_aff = nn.Linear(2048, aff_classes)

        for m in [self.fc_obj, self.fc_aff]:
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = self.gap(x)

        x = x.view(x.shape[0], -1)

        y_obj = self.fc_obj(x)
        y_aff = self.fc_aff(x)

        return [y_obj, y_aff]


class ResNet152_linearcam2(nn.Module):

    def __init__(self, obj_classes, aff_classes, pretrained=True):
        super().__init__()

        resnet50 = models.resnet152(pretrained=pretrained)
        self.feature = nn.Sequential(*list(resnet50.children())[:-2])

        self.conv_obj = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.bn_obj = nn.BatchNorm2d(2048)
        self.conv_aff = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.bn_aff = nn.BatchNorm2d(2048)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_obj = nn.Linear(2048, obj_classes)
        self.fc_aff = nn.Linear(2048, aff_classes)

        for m in list(self.children())[1:]:
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

    def forward(self, x):
        x = self.feature(x)

        y_obj = F.relu(self.bn_obj(self.conv_obj(x)))
        y_obj = self.gap(y_obj)
        y_aff = F.relu(self.bn_aff(self.conv_aff(x)))
        y_aff = self.gap(y_aff)

        y_obj = y_obj.view(x.shape[0], -1)
        y_aff = y_aff.view(x.shape[0], -1)

        y_obj = self.fc_obj(y_obj)
        y_aff = self.fc_aff(y_aff)

        return [y_obj, y_aff]
