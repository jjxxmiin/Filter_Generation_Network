import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lib.models.module import GFLayer


def get_tensor_img(path, input_shape=(32, 32)):
    img = Image.open(path)
    cvt_tensor = transforms.Compose([transforms.Resize(input_shape),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tensor_img = cvt_tensor(img).view(1, 3, input_shape[0], input_shape[1])

    return tensor_img


def scaling(img):
    img = img - np.min(img)
    img = img / np.max(img)

    return img


class GradCAM(object):
    def __init__(self, model, label, device='cuda'):
        self.model = model
        self.model.eval()

        self.device = device
        self.grads = []
        self.features = []
        self.items = []
        self.item_id = 0
        self.label = label

    def get_feature_hook(self, name):
        def hook(module, input, output):
            self.items.append('%d_%s' % (self.item_id, name))
            self.features.append(output)
            self.item_id += 1

        return hook

    def get_gradient_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0])
        print(grad_out[0].shape)

    def register(self):
        for module, (name, _) in zip(self.model.modules(), self.model.named_modules()):
            if type(module) == GFLayer or type(module) == nn.Conv2d:
                module.register_forward_hook(self.get_feature_hook(name))
                module.register_backward_hook(self.get_gradient_hook)

    def save_img(self, loader):
        tensor_img, target = next(iter(loader))
        tensor_img = tensor_img.to(self.device)
        target = target.item()

        # register hook
        self.register()

        # predict
        output = self.model(tensor_img)
        h_x = F.softmax(output, dim=1).data.squeeze()
        pred = h_x.argmax(0).item()

        one_hot_output = torch.zeros(1, h_x.size()[0])
        one_hot_output[0][target] = 1

        one_hot_output = one_hot_output.to(self.device)

        # backprop
        output.backward(gradient=one_hot_output)

        # get grad cam
        self.grads = self.grads[::-1]

        inv_normalize = transforms.Normalize(
            mean=[-0.4914 / 0.229, -0.4822 / 0.224, -0.4465 / 0.225],
            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
        )
        tensor_img = inv_normalize(tensor_img.squeeze(0))

        for idx, name in enumerate(self.items):
            grad = self.grads[idx][0].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            feature = self.features[idx][0]

            grad_cam = F.relu((grad * feature).sum(dim=0)).squeeze(0)
            scaled_grad_cam = scaling(grad_cam.detach().cpu().numpy())

            resized_grad_cam = cv2.resize(scaled_grad_cam, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * resized_grad_cam), cv2.COLORMAP_JET)

            img = tensor_img.permute(1, 2, 0).cpu().numpy() * 255.0

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (224, 224))

            cv2.imwrite(os.path.join(f'./result/origin.jpg'), img)

            heatimg = heatmap * 0.4 + img * 0.5

            cv2.imwrite(os.path.join(f'./result/{name}.jpg'), heatimg)

        print(f"Target Class Number  : {target} \n"
              f"Target Class Name    : {self.label[target]} \n"
              f"Predict Class Number : {pred} \n"
              f"Predict Class Name   : {self.label[pred]}")