import random
import os
# import wandb
# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, dataset, Subset, Dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# import models
from unlearn import *
from utils import *
import datasets
import models
import config
from training_utils import *
import os.path as osp

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, default='ResNet18', help="net type")
parser.add_argument(
        "-weight_path",
        type=str,
        default=".\log_files\model\pretrain\ResNet18-Cifar20-15\35-best.pth",
        help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
        "-dataset",
        type=str,
        default='Cifar20',
        nargs="?",
        choices=["Cifar19", "Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
        help="dataset to train on",
)
parser.add_argument("-classes", type=int, default=15, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=True, help="use gpu or not")
parser.add_argument("-b", type=int, default=1, help="batch size for dataloader")  # 128,32
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
        "-method",
        type=str,
        default="finetune",
        help="select unlearning method from choice set",
)  # not to use: "UNSIR", "ssd_tuning"?

parser.add_argument(
        "--forget_class",
        type=int,
        default=0,  # 4
        nargs="?",
        help="class to forget",
)

parser.add_argument(
        "-mia_mu_method",
        type=str,
        default="mia_mu_relearning",
        help="select unlearning method from choice set",
)

parser.add_argument(
        "-forget_perc", type=float, default=0.1, help="Percentage of trainset to forget"
)

parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
parser.add_argument("--unlearn_data_percent", type=str, default='0.1')
parser.add_argument("--relearn_lr_list", type=str, default='0.005')
parser.add_argument("--para1", type=str, default=None)
parser.add_argument("--para2", type=str, default=None)
parser.add_argument("--blackbox", type=str, default='0')
parser.add_argument("-num_classes", type=int, default=10, help="number of classes")
args = parser.parse_args()


def mia_mu_relearning(unlearned_model, forget_train_dataloader,
                      ood_dataloader, rest_retain_dataloader,
                      device, weight_path, relearn_lr=0.01):
    unlearned_model.load_state_dict(torch.load(weight_path))
    unlearned_model = unlearned_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=relearn_lr, momentum=0.9)

    def relearn(model, dataloader, relearn_weights_path=None):
        rajectory_acc_list = []
        total_loss = 0.0
        correct = 0
        total_batch = len(dataloader)
        iter_rest_retaining_dataloader = iter(rest_retain_dataloader)
        for batch_idx, (inputs, _, targets) in enumerate(tqdm(dataloader)):
            model.train()
            try:
                rest_inputs, _, rest_targets = iter_rest_retaining_dataloader.next()
            except:
                iter_rest_retaining_dataloader = iter(rest_retain_dataloader)
                rest_inputs, _, rest_targets = iter_rest_retaining_dataloader.next()

            inputs = torch.cat((inputs, rest_inputs), dim=0)
            targets = torch.cat((targets, rest_targets), dim=0)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            model.eval()
            for inputs, _, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / len(dataloader.dataset)
            accuracy = correct / len(dataloader.dataset)
            rajectory_acc_list.append(accuracy)

            total_loss = 0.0
            correct = 0
            if batch_idx > 75:
                break
        if relearn_weights_path is not None:
            torch.save(model.state_dict(), relearn_weights_path)
        return rajectory_acc_list

    unlearned_model.load_state_dict(torch.load(weight_path))
    forget_trajectory_acc_list = relearn(unlearned_model, forget_train_dataloader)#, relearn_weights_path=forget_weights_path)

    ood_trajectory_acc_list = []
    for i in range(len(ood_dataloader)):
        print(f"train on ood dataloader_{i}")
        unlearned_model.load_state_dict(torch.load(weight_path))
        ood_trajectory_acc_list.extend(relearn(unlearned_model, ood_dataloader[i]))#, relearn_weights_path=ood_weights_path))

    # Calculate the index in forget_trajectory_acc_list and ood_trajectory_acc_list
    def find_rea_index(acc_list):
        return next((i for i, acc in enumerate(acc_list) if acc >= 0.9), 75)

    rea_forget_index = find_rea_index(forget_trajectory_acc_list)
    rea_ood_index = find_rea_index(ood_trajectory_acc_list)

    return rea_forget_index, rea_ood_index

if __name__ == '__main__':
    set_seeds(args)
    batch_size = args.b

    # get network
    net = getattr(models, args.net)(num_classes=args.classes)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
        unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)
    # net.load_state_dict(torch.load(args.weight_path))

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
    img_size = 32

    trainset = getattr(datasets, args.dataset)(root=root, download=True, train=True, unlearning=True, img_size=img_size)
    validset = getattr(datasets, args.dataset)(root=root, download=True, train=False, unlearning=True, img_size=img_size)

    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)

    classwise_train, classwise_test = get_classwise_ds(trainset, num_classes=args.num_classes), \
        get_classwise_ds(validset, args.num_classes)

    if args.dataset == 'Cifar10':
        ood_classes = config.cifar10_ood_classes
    else:
        ood_classes = config.ood_classes

    (retain_train, retain_valid) = build_retain_sets_in_unlearning(classwise_train, classwise_test, args.num_classes, int(args.forget_class),
                                                     ood_classes)

    forget_train, forget_valid = classwise_train[int(args.forget_class)], classwise_test[int(args.forget_class)]

    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)

    #lets change forget_train_dl only
    len_infer_data = int(float(args.unlearn_data_percent)*len(forget_train))
    print(f"len_infer_data: {len_infer_data}")
    forget_train = Subset(forget_train, np.random.choice(range(len(forget_train)),
                                                         size=len_infer_data, replace=False))
    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    full_train_dl = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), batch_size=batch_size, )
    full_valid_dl = DataLoader(ConcatDataset((retain_valid_dl.dataset, forget_valid_dl.dataset)), batch_size=batch_size, )

    ood_valid_ds = {}
    ood_train_ds = {}
    # ood_valid_dl = []
    ood_train_dl = []
    combined_dataset_list = [forget_train]

    #only one ood class every time
    #TODO selected one ood class
    selected_ood_class = random.choice(ood_classes)
    #TODO used for plot loss landscape
    # selected_ood_class = 0
    for cls in [selected_ood_class]:
        ood_valid_ds[cls] = []
        ood_train_ds[cls] = []

        for img, label, clabel in classwise_test[cls]:
            ood_valid_ds[cls].append((img, label, int(args.forget_class)))

        for img, label, clabel in classwise_train[cls]:
            ood_train_ds[cls].append((img, label, int(args.forget_class)))

        # ood_valid_dl.append(DataLoader(ood_valid_ds[cls], batch_size=batch_size, shuffle=True))
        combined_dataset_list.append(ood_train_ds[cls])
        ood_train_dl.append(DataLoader(Subset(ood_train_ds[cls], np.random.choice(range(len(ood_train_ds[cls])), size=len_infer_data, replace=False)),
                                       batch_size=batch_size, shuffle=True))

    data_length = len(forget_train)
    retain_index_1 = np.random.choice(range(len(retain_train)), size=data_length, replace=False)
    retain_index_2 = list(set(range(len(retain_train))) - set(retain_index_1))
    retrain_train_1 = Subset(retain_train, retain_index_1)
    retrain_train_2 = Subset(retain_train, retain_index_2)
    retain_train_dl = DataLoader(retrain_train_1, num_workers=0, batch_size=batch_size, shuffle=True)
    if args.dataset == 'Cifar20' or args.dataset == 'Cifar10':
        retain_bs = 14
    else:
        retain_bs = 4 #6
    rest_retain_train_dl = DataLoader(retrain_train_2, num_workers=0, batch_size=retain_bs, shuffle=True) #TODO 14

    full_train_dataloader = DataLoader(ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), batch_size=batch_size)

    #where the unlearned model is
    checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
                                   "{unlearning_scenarios}".format(unlearning_scenarios="forget_full_class_main"),
                                   "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset, classes=args.classes),
                                   "{task}".format(task="unlearning"),
                                   "{unlearning_method}_{para1}_{para2}".format(
                                    unlearning_method=args.method,
                                    para1=args.para1,
                                    para2=args.para2))
    print("checkpoint_path_folder: {}".format(checkpoint_path_folder))
    checkpoint_path = os.path.join(checkpoint_path_folder, "{epoch}-{type}.pth")
    # weights_path = checkpoint_path.format(epoch=args.epochs, type="last")
    weights_path = checkpoint_path.format(epoch=args.forget_class, type="class_var")

    # mia_checkpoint_path_folder = os.path.join(config.CHECKPOINT_PATH,
    #                                       "{unlearning_scenarios}".format(
    #                                           unlearning_scenarios="forget_full_class_main"),
    #                                       "{net}-{dataset}-{classes}".format(net=args.net, dataset=args.dataset,
    #                                                                          classes=args.classes),
    #                                       "{task}".format(task="relearning_retain"),
    #                                       "{unlearning_method}_{para1}_{para2}".format(
    #                                           unlearning_method=args.method,
    #                                           para1=args.para1,
    #                                           para2=args.para2))
    # print("mia_checkpoint_path_folder: {}".format(mia_checkpoint_path_folder))
    # os.makedirs(mia_checkpoint_path_folder, exist_ok=True)
    # mia_checkpoint_path = os.path.join(mia_checkpoint_path_folder, "{epoch}-{type}.pth")
    # forget_weights_path = mia_checkpoint_path.format(epoch=args.epochs, type="last")
    # ood_weights_path = mia_checkpoint_path.format(epoch=args.epochs, type="ood_last")

    if args.method == 'retrain':
        net = getattr(models, args.net)(num_classes=int(args.classes-1))

    net = net.to(device)
    try:
        net.load_state_dict(torch.load(weights_path))
    except:
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            net = nn.DataParallel(net)
        net.load_state_dict(torch.load(weights_path))

    #TODO last: for cifar100: [0.001, 0.005, 0.01]
    args.relearn_lr_list = [0.001, 0.005, 0.006, 0.01, 0.012]#, 0.01, 0.03, 0.05]TODO 0.001, 0.01, 0.50, 0.1,for cifar20
    repeat_time = 1

    rea_index_list_forget_dict = {}
    rea_index_list_ood_dict = {}

    for relearn_lr in args.relearn_lr_list:
        kwargs = {
            "unlearned_model": net,
            "forget_train_dataloader": forget_train_dl,
            "ood_dataloader": ood_train_dl,
            "rest_retain_dataloader": rest_retain_train_dl,
            "device": "cuda" if args.gpu else "cpu",
            "relearn_lr": relearn_lr,
            "weight_path": weights_path,
        }

        # mia attack
        for _ in range(repeat_time):
            rea_index_forget, rea_index_ood = mia_mu_relearning(**kwargs)
            if relearn_lr not in rea_index_list_forget_dict:
                rea_index_list_forget_dict[relearn_lr] = []
            if relearn_lr not in rea_index_list_ood_dict:
                rea_index_list_ood_dict[relearn_lr] = []

            rea_index_list_forget_dict[relearn_lr].append(rea_index_forget)
            rea_index_list_ood_dict[relearn_lr].append(rea_index_ood)

    # TODO plot difference bar
    diff_dict = {}
    for lr in rea_index_list_forget_dict.keys():
        differences = [
            rea_ood - rea_forget
            for rea_forget, rea_ood in zip(rea_index_list_forget_dict[lr], rea_index_list_ood_dict[lr])
            if rea_forget is not None and rea_ood is not None
        ]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        diff_dict[lr] = (mean_diff, std_diff)

    print("rea_index_list_forget_dict", rea_index_list_forget_dict)
    print("rea_index_list_ood_dict", rea_index_list_ood_dict)