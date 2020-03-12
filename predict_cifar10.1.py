import logging
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torchvision
from torchvision import datasets, transforms

from models.cifar.resnet import resnet
from models.cifar.wrn import wrn
from py3simplex import plotSimplex

logger = logging.getLogger(__name__)


class CIFAR10_1(data.Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, i):
        image = self.images[i]
        image = self.transform(image)
        label = self.labels[i]
        return image, label

    def __len__(self):
        return self.labels.shape[0]


def main():

    logger.info('Loading cifar-10.1 data')
    image_array = np.load('data/cifar10.1_v6_data.npy')
    label_array = np.load('data/cifar10.1_v6_labels.npy')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    inv_transform = transforms.Normalize((-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
                                         (1 / 0.2023, 1 / 0.1994, 1/ 0.2010))

    dataset = CIFAR10_1(image_array, label_array, transform)
    dataloader = data.DataLoader(dataset, batch_size=1000, shuffle=False)

    logger.info('Loading model')
    model = resnet(num_classes=10, depth=110)
    # model = wrn(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
    model = torch.nn.DataParallel(model).cuda()
    # checkpoint = torch.load('resnet-110/model_best.pth.tar')
    checkpoint = torch.load('resnet-110/cifar10.pth.tar')
    # checkpoint = torch.load('./checkpoint/WRN-28-10-drop/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    i = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.max(probs, 1)
        for target, logit in zip(targets, logits):
            tgt_string = '%i ' % target.item()
            prediction_strings = ['%0.8f' % x for x in logit.tolist()]
            print(tgt_string + ' '.join(prediction_strings))
        # incorrect = indices != targets
        # high_confidence = values > 0.90
        # idx = incorrect & high_confidence

        # misses = inputs[idx]
        # y_hats = indices[idx]
        # ys = targets[idx]
        # confs = values[idx]

        # import pdb; pdb.set_trace()
        # preds_0 = probs[:,0].cpu().numpy()
        # preds_1 = probs[:,1].cpu().numpy()
        # preds_2 = probs[:,2:].sum(dim=1).cpu().numpy()
        # points = np.vstack([preds_0, preds_1, preds_2]).T
        # colors = targets.cpu().numpy()
        # colors[colors>1] = 2

        # plotSimplex(points, c=colors)
        # plt.savefig('simplex.png')


        # for tensor, y, y_hat, conf in zip(inputs, targets, indices, values):
        #     image = inv_transform(tensor)
        #     image.squeeze_()
        #     torchvision.utils.save_image(image, filename='img/%i.png' % i)
        #     correct = labels[y.item()]
        #     predicted = labels[y_hat.item()]
        #     print('%i %s %s %0.4f' % (i, correct, predicted, conf))
        #     i+=1


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    main()

