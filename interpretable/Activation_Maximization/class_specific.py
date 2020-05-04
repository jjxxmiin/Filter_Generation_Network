import os
import copy
import numpy as np
import torch
from torchvision import models
from PIL import Image


def preprocess_image(img):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    im_as_arr = np.float32(img)
    im_as_arr = im_as_arr.transpose(2, 0, 1)

    # 채널 정규화
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    # tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = torch.autograd.Variable(im_as_ten, requires_grad=True)

    return im_as_var


def recreate_image(im_as_var):
    reverse_mean = [-0.4914, -0.4822, -0.4465]
    reverse_std = [1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    return recreated_im


def save_image(im, path):
    if isinstance(im, (np.ndarray, np.generic)):
        if np.max(im) <= 1:
            im = (im*255).astype(np.uint8)
        im = Image.fromarray(im)
    im.save(path)


if not os.path.exists('./generated'):
    os.makedirs('./generated')

DEVICE = 'cuda'

img = Image.fromarray(np.uint8(np.random.uniform(0, 255, (224, 224, 3))))

model = models.vgg16(pretrained=True).to(DEVICE)
model.eval()

target_class = 8
initial_learning_rate = 20
created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

for i in range(1, 150):
    processed_image = preprocess_image(created_image)
    optimizer = torch.optim.SGD([processed_image], lr=initial_learning_rate)
    output = model(processed_image.to(DEVICE))
    class_loss = -output[0, target_class]

    print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.cpu().data.numpy()))
    model.zero_grad()
    class_loss.backward()

    # Update image
    optimizer.step()

    # Recreate image
    created_image = recreate_image(processed_image)

    if i % 10 == 0:
        # Save image
        im_path = './generated/c_specific_iteration_' + str(i) + '.jpg'
        save_image(created_image, im_path)
