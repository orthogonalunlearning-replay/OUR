import os
import os.path as osp
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import datasets
import models
import config
from datasets import transform_imagenet
from training_utils import *
from utils import *

# from utils import get_classwise_ds, get_imagenet_classwise_ds

# Original code from https://github.com/weiaicunzai/pytorch-cifar100 <- refer to this repo for comments

# Create datasets of the classes

# Creates datasets for method execution
def build_retain_sets(classwise_train, classwise_test, num_classes, ood_class):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    retain_class = list(set(all_class) - set(ood_class))

    retain_valid = []
    retain_train = []

    for ordered_cls, cls in enumerate(retain_class):
        for img, label in classwise_test[cls]: #label and coarse label
            retain_valid.append((img, ordered_cls)) #ordered_clss

        for img, label in classwise_train[cls]:
            retain_train.append((img, ordered_cls))

    return (retain_train, retain_valid)

def train(epochs):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(trainloader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch <= args.warm:
            warmup_scheduler.step()


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    net.eval()

    train_loss = 0.0  # cost function error
    train_correct = 0.0

    for images, labels in trainloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        train_loss += loss.item()
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print("Test set: Epoch: {}, Train Average loss: {:.4f}, Train Accuracy: {:.4f}, Average loss: {:.4f}, Accuracy: {:.4f}".format(
            epoch,
            train_loss / len(trainloader.dataset),
            train_correct.float() / len(trainloader.dataset),
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
        )
    )
    print()

    return correct.float() / len(testloader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, default='wrn_28_2', help="net type")
    parser.add_argument("-dataset", type=str, default='Imagenet64', help="dataset to train on: Cifar10, Cifar20, Cifar100")
    parser.add_argument("-num_classes", type=int, default=100, help="number of original classes")
    parser.add_argument("-classes", type=int, default=94, help="number of classes")
    parser.add_argument("-gpu", default=True, help="use gpu or not")
    parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("-seed", type=int, default=0, help="seed for runs")
    args = parser.parse_args()

    MILESTONES = (
        getattr(config, f"{args.dataset}_MILESTONES")
        if args.net != "ViT"
        else getattr(config, f"{args.dataset}_ViT_MILESTONES")
    )
    EPOCHS = (
        getattr(config, f"{args.dataset}_EPOCHS")
        if args.net != "ViT"
        else getattr(config, f"{args.dataset}_ViT_EPOCHS")
    )
    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    # multi-gpu
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    if args.gpu:
        net = net.cuda()

    # dataloaders
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = 224 if args.net == "ViT" else 32
    # trainset = getattr(datasets, args.dataset)(root=root, train=True, transform=transform_imagenet)
    # validset = getattr(datasets, args.dataset)(root=root, train=False, transform=transform_imagenet)
    #
    # classwise_train, classwise_test = get_imagenet_classwise_ds(trainset, num_classes=args.num_classes), \
    #                                   get_imagenet_classwise_ds(validset, num_classes=args.num_classes)
    #
    # #changet the label
    # #it should be satteled
    # (retain_train, retain_valid) = build_retain_sets(classwise_train, classwise_test, args.num_classes, config.ood_classes)

    # traindir = os.path.join(root, 'Imagenet64/train')
    # valdir = os.path.join(root, 'Imagenet64/val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.482, 0.458, 0.408],
                                     std=[0.269, 0.261, 0.276])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    trainset = getattr(datasets, args.dataset)(root=osp.join(root, 'Imagenet64_subset-100'), train=True)
    validset = getattr(datasets, args.dataset)(root=osp.join(root, 'Imagenet64_subset-100'), train=False)

    classwise_train, classwise_test = get_classwise_ds_gtsrb(trainset, num_classes=args.num_classes), \
        get_classwise_ds_gtsrb(validset, num_classes=args.num_classes)

    (retain_train, retain_valid) = build_retain_sets(classwise_train, classwise_test, args.num_classes,
                                                     config.ood_classes)

    trainloader = DataLoader(retain_train, batch_size=args.b, shuffle=True)
    testloader = DataLoader(retain_valid, batch_size=args.b, shuffle=False)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2)  # learning rate decay
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "{task}".format(task="pretrain"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:  # and epoch > MILESTONES[1]
            weights_path = checkpoint_path.format(
                epoch=epoch, type="best"
            )
            print("saving weights file to {}".format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc

