#!/bin/python3.8

"""
This file is used to collect all arguments for the experiment, prepare the dataloaders, call the method for forgetting, and gather/log the metrics.
Methods are executed in the strategies file.
"""

import random
import os

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import ScalarFormatter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import models
from pretrain_model_cifar15 import get_classwise_ds
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import models
import config
from training_utils import *
import os.path as osp
from scipy.spatial.distance import pdist, squareform

@torch.no_grad()
def eval_training(net,testloader):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    return correct.float() / len(testloader.dataset)

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    default="./log_files/model/pretrain/ResNet18-Cifar20-19/39-best.pth",
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    default="Cifar20",
    nargs="?",
    choices=["Cifar10", "Cifar19", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, default=15, help="number of classes")
parser.add_argument("-gpu", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    default="baseline",
    nargs="?",
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_class",
    type=str,
    default="4",  # 4
    nargs="?",
    help="class to forget"
)

parser.add_argument(
    "-mia_mu_method",
    type=str,
    default="mia_mu_adversarial",
    nargs="?",
    help="select unlearning method from choice set",
) #not to use: "UNSIR", "ssd_tuning"

parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")

parser.add_argument(
    "--para1", type=str, default=None, help="the first parameters, lr, etc."
)
parser.add_argument(
    "--para2", type=str, default=None, help="number of epochs"
)

#############masked related##########################
parser.add_argument("--masked_path", default=None, help="the path where masks are saved")

args = parser.parse_args()

def extract_features(dataloader, model_temp, device):
    features = []
    labels = []
    model_temp.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, _, label = data
            inputs = inputs.to(device)
            output = model_temp(inputs)
            features.extend(output.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

# executes the method passed via args
def tsne_visualization(train_loader, model, forget_class, device, name, seed=42):

    features, labels = extract_features(train_loader, model, device)  # full_trainloader #train_dl_w_ood
    tsne = TSNE(n_components=2, random_state=seed)
    reduced_features = tsne.fit_transform(features) #[class]
    print("reduced_features", reduced_features.shape)
    # retain_classes = list(set(range(20)) - set(config.ood_classes))
    # # print("retain_classes", retain_classes)
    # index_forget_class = retain_classes.index(forget_class)

    distances = pdist(reduced_features, 'sqeuclidean')
    dist_matrix = squareform(distances)
    inv_distances = 1 / (1 + dist_matrix)
    np.fill_diagonal(inv_distances, 0)
    Q = inv_distances / np.sum(inv_distances)

    return Q, reduced_features, labels

def kl_divergence(P, Q):
    P_safe = np.where(P == 0, np.finfo(float).eps, P)
    Q_safe = np.where(Q == 0, np.finfo(float).eps, Q)

    return np.sum(P_safe * np.log(P_safe / Q_safe))

def feature_visualization_metric(unlearned_model, device):
    return tsne_visualization(full_train_dl, unlearned_model,  int(args.forget_class), device, name='unlearned_model', seed=42)
    
def compute_centroid(points):
    return np.mean(points, axis=0)

def class_to_overall_relationship(Y, labels):
    unique_labels = np.unique(labels)
    overall_centroid = compute_centroid(Y)
    centroids = {label: compute_centroid(Y[labels == label]) for label in unique_labels}
    distances = {label: np.linalg.norm(centroid - overall_centroid) for label, centroid in centroids.items()}
    return distances


import numpy as np

def calculate_class_variances(Y, labels, unlearned_label=4):
    unique_labels = np.unique(labels)
    overall_centroid = np.mean(Y, axis=0)
    class_centroids = np.array([np.mean(Y[labels == label], axis=0) for label in unique_labels])

    intra_class_variances = {
        label: np.mean(np.sum((Y[labels == label] - class_centroids[i]) ** 2, axis=1))
        for i, label in enumerate(unique_labels)
    }

    inter_class_variance = np.sum((class_centroids - overall_centroid) ** 2, axis=1).mean()

    return intra_class_variances, inter_class_variance

def calculate_density(Y, labels):
    unique_labels = np.unique(labels)
    densities = {}
    for label in unique_labels:
        class_points = Y[labels == label]
        volume = np.max(np.linalg.norm(class_points - np.mean(class_points, axis=0), axis=1))**2
        densities[label] = len(class_points) / volume if volume != 0 else 0
    return densities

from scipy.stats import gaussian_kde

def estimate_overlap(Y, labels):
    unique_labels = np.unique(labels)
    kdes = {label: gaussian_kde(Y[labels == label].T) for label in unique_labels}
    grid_points = np.linspace(np.min(Y), np.max(Y), 100)
    grid_mesh = np.meshgrid(grid_points, grid_points)
    grid_coords = np.vstack([grid.ravel() for grid in grid_mesh])

    densities = np.array([kde(grid_coords) for kde in kdes.values()])
    overlap = np.min(densities, axis=0).sum()
    return overlap

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def calculate_overlap(Y, labels, target_label, save_path):
    target_data = Y[labels == target_label]
    other_data = Y[labels != target_label]

    target_kde = gaussian_kde(target_data.T)
    other_kde = gaussian_kde(other_data.T)

    x_min, x_max = Y[:, 0].min() - 1, Y[:, 0].max() + 1
    y_min, y_max = Y[:, 1].min() - 1, Y[:, 1].max() + 1
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    target_density = target_kde(positions).reshape(xx.shape)
    other_density = other_kde(positions).reshape(xx.shape)

    overlap = np.minimum(target_density, other_density)

    plt.figure(figsize=(4.1, 3.5))
    cax=plt.imshow(np.rot90(overlap), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
    # plt.title('Overlap Area between Class 4 and Others')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.colorbar()
    # plt.show()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    plt.colorbar(cax, format=formatter)
    plt.savefig(save_path + f"/kde_overlap.pdf", bbox_inches='tight', format='pdf')

    return overlap.sum()

def plot_tsne_visualization_results(reduced_features, forget_class=4, name='badteacher'):
    retain_classes = list(set(range(20)) - set(config.ood_classes))
    # print("retain_classes", retain_classes)
    index_forget_class = retain_classes.index(forget_class)
    plt.figure(figsize=(5.1, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
              "#d62728", "#8c564b", "#e377c2", "#7f7f7f",
              "#bcbd22", "#17becf", "#65c2a4", "#89a6af",
              "#df7976", "#e4d1dd", "#d1e0ab"]
    for i in list(set(range(15)) - set([index_forget_class])):
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], color=colors[i],
                    label=str(retain_classes[i]), alpha=0.5)

    for i in set([index_forget_class]):
        indices = labels == i
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], color=colors[i],
                    label=str(retain_classes[i]), alpha=0.5)

    plt.legend(loc=4)
    plt.savefig(visual_path + f"/tsne_{name}.pdf", bbox_inches='tight', format='pdf')


from sklearn.metrics import silhouette_score

if __name__ == '__main__':

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    batch_size = args.b
    Q = []

    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    if args.para1:
        q_save_path = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(
                                       unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                                                                                para1=args.para1, para2=args.para2))

        checkpoint_path = os.path.join(config.CHECKPOINT_PATH,
                                       "{unlearning_scenarios}".format(
                                           unlearning_scenarios="forget_full_class_main"),
                                       "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                          classes=args.classes),
                                       "{task}".format(task="unlearning"),
                                       "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                                                                                    para1=args.para1,
                                                                                    para2=args.para2))
    else:
        q_save_path = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(
                                       unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}".format(unlearning_method=args.method))

        checkpoint_path = os.path.join(config.CHECKPOINT_PATH,
                                       "{unlearning_scenarios}".format(
                                           unlearning_scenarios="forget_full_class_main"),
                                       "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                          classes=args.classes),
                                       "{task}".format(task="unlearning"),
                                       "{unlearning_method}".format(unlearning_method=args.method))

    print("#####", checkpoint_path)

    weights_path = os.path.join(checkpoint_path, "{epoch}-{type}.pth").format(epoch=args.epochs, type="last")

    # For celebritiy faces
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    img_size = 224 if args.net == "ViT" else 32

    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True,
                                               img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True,
                                               img_size=img_size)

    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)

    classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=20), \
                                      get_classwise_ds(validset, num_classes=20)

    (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, 20,
                                                                   int(args.forget_class), config.ood_classes)

    forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]

    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)
    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=False)

    full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
                               batch_size=batch_size, )


    full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)),
                               batch_size=batch_size, )

    ood_valid_ds = {}
    ood_train_ds = {}
    ood_valid_dl = []
    ood_train_dl = []
    for cls in config.ood_classes:
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(args.forget_class)))

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(args.forget_class)))

        ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size))
        ood_train_dl.append(DataLoader(ood_train_ds[cls], batch_size))

    train_dl_w_ood = DataLoader(ConcatDataset((retain_train_dl.dataset, ood_train_dl[0].dataset)),
                                batch_size=batch_size, shuffle=True)
    # Change alpha here as described in the paper
    # For PinsFaceRe-cognition, we use α=50 and λ=0.1
    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 0.5
    else:
        model_size_scaler = 1

    unlearned_net = getattr(models, args.net)(num_classes=args.classes)
    unlearned_net.load_state_dict(torch.load(weights_path))
    if args.gpu:
        unlearned_net = unlearned_net.cuda()

    kwargs = {
        "unlearned_model": unlearned_net,
        "device": "cuda" if args.gpu else "cpu",
    }

    if args.para1:
        visual_path = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes),
                                   "{task}".format(task="visualization"),
                                   "{unlearning_method}-{para1}-{para2}".format(unlearning_method=args.method,
                                                                                para1=args.para1,
                                                                                para2=args.para2))
    else:
        visual_path = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
                                                                      classes=args.classes),
                                   "{task}".format(task="visualization"),
                                   "{unlearning_method}".format(unlearning_method=args.method))

    print("#####", visual_path)
    if not os.path.exists(visual_path):
       os.makedirs(visual_path)
    
    Q, reduced_features, labels = feature_visualization_metric(**kwargs)
    np.save(osp.join(q_save_path, 'q.npy'), Q)
    np.save(osp.join(q_save_path, 'reduced_features.npy'), reduced_features)
    np.save(osp.join(q_save_path, 'labels.npy'), labels)
    
    plot_tsne_visualization_results(reduced_features, name=args.method)

    reduced_features = np.load(osp.join(q_save_path, 'reduced_features.npy'))
    labels = np.load(osp.join(q_save_path, 'labels.npy'))

    intra_variances, inter_variance = calculate_class_variances(reduced_features, labels,
                                                                unlearned_label=4)
    print("Intra-class Variances:", intra_variances[4])


    overlap_measure = calculate_overlap(reduced_features, labels, 4, q_save_path)
    print("Overlap Measure:", overlap_measure)


    unlearned_class = 4
    target_indices = labels == unlearned_class

    unique_labels = np.unique(labels)
    distances_to_other_classes = []
    for label in unique_labels:
        if label != unlearned_class:
            inter_class_distance = np.mean(np.linalg.norm(
                reduced_features[labels == unlearned_class, :] - reduced_features[labels == label, :].mean(axis=0),
                axis=1
            ))
            distances_to_other_classes.append((inter_class_distance, label))

    _, nearest_class = min(distances_to_other_classes, key=lambda x: x[0])
    nearest_class_indices = labels == nearest_class

    subset_indices = target_indices | nearest_class_indices
    subset_data = reduced_features[subset_indices]
    subset_labels = labels[subset_indices]

    score = silhouette_score(subset_data, subset_labels)
    print("Silhouette Score for class {} (including its nearest neighbor class): {}".format(unlearned_class, score))
