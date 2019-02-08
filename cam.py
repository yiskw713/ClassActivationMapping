import torch
import torch.nn as nn
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


""" Class Activation Mapping """


class CAM(object):
    def __init__(self, model, target_layer_obj, target_layer_aff):
        """
        Args:
            model: ResNet_linear()
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model
        self.target_layer_obj = target_layer_obj
        self.target_layer_aff = target_layer_aff

        # save values of activations and gradients in target_layer
        self.values_obj = SaveValues(self.target_layer_obj)
        self.values_aff = SaveValues(self.target_layer_aff)

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
                    [{obj_id: cam}, {aff_id1: cam1, aff_id2: cam2, ...}]
        """

        pred_obj, pred_aff = self.model(x)

        # object classification
        pred_obj = torch.sigmoid(pred_obj)
        pred_obj[pred_obj > 0.5] = 1
        pred_obj[pred_obj <= 0.5] = 0

        # affordance classification
        pred_aff = torch.sigmoid(pred_aff)
        pred_aff[pred_aff > 0.5] = 1
        pred_aff[pred_aff <= 0.5] = 0

        print("predicted object ids {}".format(pred_obj))
        print("predicted affordance ids {}".format(pred_aff))

        weight_fc_obj = list(self.model._modules.get('fc_obj').parameters())[0].to('cpu').data
        weight_fc_aff = list(self.model._modules.get('fc_aff').parameters())[0].to('cpu').data

        cams_obj = dict()
        cams_aff = dict()

        for i in pred_obj.nonzero():
            cam = self.getCAM(self.values_obj, weight_fc_obj, i)
            cams_obj[i[1].item()] = cam    # i[i] is object id

        for i in pred_aff.nonzero():
            cam = self.getCAM(self.values_aff, weight_fc_aff, i)
            cams_aff[i[1].item()] = cam.data    # i[i] is affordance id

        return cams_obj, cams_aff

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, index):
        '''
        values: the activations and gradients of target_layer
        activations: feature map before GAP.  shape => (N, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        cam: class activation map.  shape=> (N, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        cam = cam[index[0], index[1], :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


""" Grad CAM """


class GradCAM(CAM):
    """
    Args:
        model: ResNet_linear()
        target_layer: conv_layer before Global Average Pooling
    """

    def __init__(self, model, target_layer_obj, target_layer_aff):
        super().__init__(model, target_layer_obj, target_layer_aff)

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
                    [{obj_id: cam}, {aff_id1: cam1, aff_id2: cam2, ...}]
        """

        score_obj, score_aff = self.model(x)

        # object classification
        pred_obj = torch.sigmoid(score_obj)
        pred_obj[pred_obj > 0.5] = 1
        pred_obj[pred_obj <= 0.5] = 0

        # affordance classification
        pred_aff = torch.sigmoid(score_aff)
        pred_aff[pred_aff > 0.5] = 1
        pred_aff[pred_aff <= 0.5] = 0

        print("predicted object ids {}".format(pred_obj))
        print("predicted affordance ids {}".format(pred_aff))

        cams_obj = dict()
        cams_aff = dict()

        # caluculate cam of each predicted class
        for i in pred_obj.nonzero():
            cam = self.getGradCAM(self.values_obj, score_obj, i)
            cams_obj[i[1].item()] = cam

        for i in pred_aff.nonzero():
            cam = self.getGradCAM(self.values_aff, score_aff, i)
            cams_aff[i[1].item()] = cam

        return cams_obj, cams_aff

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, index):
        self.model.zero_grad()
        score[index[0], index[1]].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        cam = (alpha * activations).sum(dim=1, keepdim=True)    # shape => (1, 1, H', W')
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(GradCAM):
    """
    Args:
        model: ResNet_linear()
        target_layer: conv_layer before Global Average Pooling
    """

    def __init__(self, model, target_layer_obj, target_layer_aff):
        super().__init__(model, target_layer_obj, target_layer_aff)

    def forward(self, x):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
                    [{obj_id: cam}, {aff_id1: cam1, aff_id2: cam2, ...}]
        """

        score_obj, score_aff = self.model(x)

        # object classification
        pred_obj = torch.sigmoid(score_obj)
        pred_obj[pred_obj > 0.5] = 1
        pred_obj[pred_obj <= 0.5] = 0

        # affordance classification
        pred_aff = torch.sigmoid(score_aff)
        pred_aff[pred_aff > 0.5] = 1
        pred_aff[pred_aff <= 0.5] = 0

        print("predicted object ids {}".format(pred_obj))
        print("predicted affordance ids {}".format(pred_aff))

        cams_obj = dict()
        cams_aff = dict()

        # caluculate cam of each predicted class
        for i in pred_obj.nonzero():
            cam = self.getGradCAMpp(self.values_obj, score_obj, i)
            cams_obj[i[1].item()] = cam

        for i in pred_aff.nonzero():
            cam = self.getGradCAMpp(self.values_aff, score_aff, i)
            cams_aff[i[1].item()] = cam

        return cams_obj, cams_aff

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, index):
        self.model.zero_grad()
        score[index[0], index[1]].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        denominator += (activations * gradients.pow(3)).view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[index[0], index[1]].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        cam = (weights * activations).sum(1, keepdim=True)    # shape => (1, 1, H', W')
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data
