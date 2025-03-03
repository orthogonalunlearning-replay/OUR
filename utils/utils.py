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
    images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    return loss


def validation_step(model, batch, device):
    images, labels, clabels = batch   # labels 100, clabels 20
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
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
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(
    epochs, model, train_loader, val_loader, device, lr=0.01, milestones=None, mask=None
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
        for batch in train_loader:
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

        # Validation phase
        result = evaluate(model, val_loader, device)
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
            for img, label, clabel in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, label, ordered_cls))  # ordered_clss

            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, ordered_cls))

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
        if ordered_cls != index_of_forget_class:
            # print("ordered_cls", ordered_cls)
            for img, label in classwise_test[cls]:  # label and coarse label
                retain_valid.append((img, ordered_cls))  # ordered_clss

            for img, label in classwise_train[cls]:
                retain_train.append((img, ordered_cls))

    return (retain_train, retain_valid)

def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        # print("label", label, "clabel, ", clabel)
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds

def get_classwise_ds_gtsrb(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []
    for img, label in ds:
        classwise_ds[label].append((img, int(label)))
    return classwise_ds


