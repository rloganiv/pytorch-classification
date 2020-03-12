import logging
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import datasets, transforms, models

logger = logging.getLogger(__name__)


def main():

    logger.info('Loading imagenet data')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # inv_transform = transforms.Normalize((-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
    #                                      (1 / 0.2023, 1 / 0.1994, 1/ 0.2010))

    sys.stdout = open(os.devnull, 'w')
    # dataset = datasets.ImageNet(root='data/', split='val', download=True,
    #                             transform=transform)
    dataset = datasets.ImageFolder(root='data/imagenetv2-matched-frequency',
                                   transform=transform)
    # dataset = datasets.ImageFolder(root='data/imagenetv2-threshold0.7',
    #                                transform=transform)
    # dataset = datasets.ImageFolder(root='data/imagenetv2-topimages',
    #                                transform=transform)
    sys.stdout = sys.__stdout__
    dataloader = data.DataLoader(dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=4)

    logger.info('Loading model')
    model = models.resnet152(pretrained=True)
    # model = models.resnext101_32x8d(pretrained=True)

    model = model.cuda()
    model.eval()

    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            logits = model(inputs)

        for target, logit in zip(targets, logits):
            tgt_string = '%i ' % target.item()
            prediction_strings = ['%0.8f' % x for x in logit.tolist()]
            print(tgt_string + ' '.join(prediction_strings))

        # Adversarial images

        # for i in range(25):
        #     inputs.requires_grad = True
        #     logits = model(inputs)
        #     logp = F.log_softmax(logits, -1)
        #     loss = logp[:,100].sum()
        #     loss.backward()
        #     inputs.grad
        #     delta = F.normalize(inputs.grad)
        #     inputs = (inputs + 0.1 * delta).detach()
        #     inputs = F.normalize(inputs)

        # for i, image in enumerate(inputs):
        #     torchvision.utils.save_image(image, filename=f'trip{i}.png')

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

