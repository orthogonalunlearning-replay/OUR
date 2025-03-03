# From https://github.com/vikram2000b/bad-teaching-unlearning
import csv

import torch
from torch import nn
from torch.nn import functional as F
from training_utils import *
import numpy as np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100


def training_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss


def validation_step(model, batch, device):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    out = model(images)  # Generate predictions
    # print('out', out)
    # print("labels", labels)
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,
            result["lrs"][-1],
            result["train_loss"],
            result["Loss"],
            result["Acc"],
        )
    )

@torch.no_grad()
def evaluate_gtsrb(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle_gtsrb(
    epochs, model, train_loader, val_loader, device, forget_loader=None, lr=0.01, milestones=None, mask=None
):
    torch.cuda.empty_cache()
    history = []

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.8
        )  # learning rate decay
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))

    for epoch in range(epochs):
        if epoch > 1 and milestones:
            train_scheduler.step(epoch)

        model.train()
        train_losses = []
        lrs = []
        for idx, batch in enumerate(train_loader):
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()

            if mask:
                # print("finetune's mask is available!")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if epoch <= 1 and milestones:
                warmup_scheduler.step()

            # if idx % 10==0:
            #     with torch.no_grad():
            #         model.eval()
            #         print(f"Epoch[{epoch}]:", "Retain Dataset Acc",
            #               evaluate_gtsrb(model, val_loader, next(model.parameters()).device), ",Forget Dataset Acc",
            #               evaluate_gtsrb(model, forget_loader, next(model.parameters()).device))
            #         forget_acc = evaluate_gtsrb(model, forget_loader, next(model.parameters()).device)["Acc"]
            #         if forget_acc < 0.3:
            #             return

        # Validation phase
        result = evaluate_gtsrb(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                data.append([float(x) for x in row[0].split('\t')])
    return np.asarray(data)

def build_retain_sets_in_unlearning(classwise_train, classwise_test, num_classes, forget_class, ood_class):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    retain_class = list(set(all_class) - set(ood_class))

    retain_valid = []
    retain_train = []

    assert forget_class in retain_class
    index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        if ordered_cls !=index_of_forget_class:
            for img, label in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, ordered_cls))  # ordered_clss

            for img, label in classwise_train[cls]:
                retain_train.append((img, ordered_cls))

    return (retain_train, retain_valid)

def build_retain_sets_in_unlearning_gtsrb(classwise_train, classwise_test, num_classes, forget_class, ood_class):
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    retain_class = list(set(all_class) - set(ood_class))

    retain_valid = []
    retain_train = []

    assert forget_class in retain_class
    index_of_forget_class = retain_class.index(forget_class)

    for ordered_cls, cls in enumerate(retain_class):
        if ordered_cls !=index_of_forget_class:
            for img, label in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, ordered_cls))  # ordered_clss

            for img, label in classwise_train[cls]:
                retain_train.append((img, ordered_cls))

    return (retain_train, retain_valid)

def fit_one_unlearning_cycle_gtsrb(epochs, model, train_loader, val_loader, lr, device, forget_loader=False, mask=None, dataname='GTSRB'):
    history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for idx, batch in enumerate(train_loader):
            loss = training_step(model, batch, device)
            loss.backward()
            train_losses.append(loss.detach().cpu())

            if mask:
                # print("amnesiac's mask is available!")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            if idx % 10 == 0 and forget_loader is not None:
                with torch.no_grad():
                    model.eval()
                    print(f"Epoch[{epoch}]:", "Retain Dataset Acc",
                          evaluate_gtsrb(model, val_loader, next(model.parameters()).device), ",Forget Dataset Acc",
                          evaluate_gtsrb(model, forget_loader, next(model.parameters()).device))
                    forget_acc = evaluate_gtsrb(model, forget_loader, next(model.parameters()).device)["Acc"]
                    if forget_acc < 1:
                        return

        result = evaluate_gtsrb(model, val_loader, device)
        result["train_loss"] = torch.stack(train_losses).mean()
        result["lrs"] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
    return history