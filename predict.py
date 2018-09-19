import logging

import torch
from torch.utils import data
from torchvision import datasets, transforms

from models.cifar.resnet import resnet

logger = logging.getLogger(__name__)


def main():
    logger.info('Loading cifar-110 test data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR100(root='data/', train=False, download=True,
                                transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False,
                                 num_workers=4)

    logger.info('Loading model')
    model = resnet(num_classes=100, depth=110)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('resnet-110/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
        for target, prob in zip(targets, probs):
            tgt_string = '%i ' % target.item()
            prediction_strings = ['%0.8f' % x for x in prob.tolist()]
            print(tgt_string + ' '.join(prediction_strings))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    main()

