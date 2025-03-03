"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate subclass forgetting.
Seperate file to allow for easy reuse.
"""


import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset

from sklearn import linear_model, model_selection
from tqdm import tqdm

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import config


def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[label].append((img, label, clabel))
    return classwise_ds

def build_retain_sets(classwise_train, classwise_test, num_classes, forget_class, ood_class): # num_classes = 100
    # Getting the retain validation data
    all_class = list(range(0, num_classes))
    retain_class = list(set(all_class) - set([forget_class]) - set(ood_class))

    retain_valid = []
    for cls in retain_class:
        for img, label, clabel in classwise_test[cls]:
            retain_valid.append((img, label, clabel))

    retain_train = []
    for cls in retain_class:
        for img, label, clabel in classwise_train[cls]:
            retain_train.append((img, label, clabel))

    return (retain_train, retain_valid)

def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, label, clabel))

    retain_valid = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, label, clabel))

    forget_train = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, label, clabel))

    retain_train = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)


def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    # loss_acc_dict = evaluate(model, valid_dl, device)
    # retain_acc_dict = evaluate(model, retain_valid_dl, device)
    # zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    # d_f = evaluate(model, forget_valid_dl, device)
    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    #
    # return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, d_f["Acc"])

    loss_acc_dict = evaluate(model, valid_dl, device)
    d_f_acc_dict = evaluate(model, forget_train_dl, device)
    retain_acc_dict = evaluate(model, retain_train_dl, device)
    zrf = UnLearningScore(model, unlearning_teacher, forget_train_dl, 128, device)
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    return loss_acc_dict["Acc"], d_f_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia


def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(config, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(config, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(config, f"{dataset_name}_EPOCHS")
        milestones = getattr(config, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def badteacher(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset)))

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    badteacher_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def badteacher_with_prune(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl, forget_valid_dl, valid_dl, device, **kwargs,):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001) #lr=0.0001,
    retain_train_subset = random.sample(retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset)))

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    unlearning_teacher = deepcopy(unlearning_teacher)
    full_trained_teacher = deepcopy(model)
    retain_data = retain_train_subset
    forget_data = forget_train_dl.dataset
    batch_size = b_s

    # creating the unlearning dataset.
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    student_model.train()

    ########
    with_l1 = True
    no_l1_epochs = 0  # "non l1 epochs"
    unlearn_lr = 0.01  # "initial learning rate"
    unlearn_epochs = 1  # "number of total epochs for unlearn to run"
    alpha = 5e-5  # "unlearn noise"

    for epoch in range(unlearn_epochs):
        ########
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        losses = []
        for batch in unlearning_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                full_teacher_logits = full_trained_teacher(x)
                unlearn_teacher_logits = unlearning_teacher(x)
            output = student_model(x)
            optimizer.zero_grad()
            loss = UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits, unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature,)

            ########
            if epoch < unlearn_epochs - no_l1_epochs:
                current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))  # decaying
                ## current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))  # increasing
            elif unlearn_epochs - no_l1_epochs == 0:
                current_alpha = alpha
            else:
                current_alpha = 0
            if with_l1:
                loss += current_alpha * l1_regularization(student_model)

            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))


    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    forget_superclass,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_superclasses))   # 因为ood为后5个superclass，所以是按顺序去掉的这里不用改
    unlearninglabels.remove(forget_superclass)

    unlearning_trainset = []

    for x, y, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, y, random.choice(unlearninglabels)))

    for x, y, clabel in retain_train_dl.dataset:
        unlearning_trainset.append((x, y, clabel))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_superclasses:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_subclasses,
    forget_subclass,
    forget_superclass,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        num_subclasses,
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_subclasses):
        if i != forget_subclass:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_superclass
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 250, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise, forget_class_label, retain_samples, noise_batch_size, device=device
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = fit_one_unlearning_cycle(
        1, model, heal_loader, retain_valid_dl, device=device, lr=0.0001
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def pdr_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)
    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

from collections import defaultdict
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

from itertools import cycle
def train_negrad_1(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, alpha):
    model.train()

    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r, _, target_r = batch_retain
        input_f, _, target_f = batch_forget
        output_r = model(input_r)
        output_f = model(input_f)
        loss = alpha * (loss_fn(output_r, target_r) +
                        l2_penalty(model, model_init, weight_decay=0.1)) - \
               (1 - alpha) * loss_fn(output_f, target_f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def train_negrad(epoch, train_loader, delete_loader, model, criterion, optimizer, alpha):
    """vanilla training"""
    model.train()
    quiet = False
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # end = time.time()
    for idx, ((input, target), (del_input, del_target)) in enumerate(zip(train_loader, cycle(delete_loader))):
        #del_input, del_target = next(cycle(delete_loader))
        # data_time.update(time.time() - end)

        input = input.float()
        del_input = del_input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            del_input = del_input.cuda()
            del_target = del_target.cuda()

        # ===================forward=====================
        output = model(input)
        del_output = model(del_input)
        r_loss = criterion(output, target)
        del_loss = criterion(del_output, del_target)

        loss = alpha*r_loss - (1-alpha)*del_loss

        if not quiet:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # ===================meters=====================
        # batch_time.update(time.time() - end)
        # end = time.time()

    return top1.avg, losses.avg

import copy
import torch.nn as nn
def negative_grad(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    alpha = 0.95
    epochs = 10
    lr = 0.01
    quiet = True
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_1(model, model_init, retain_train_dl, forget_train_dl, loss_fn, optimizer, alpha)
        # train_negrad(epoch, retain_train_dl, forget_train_dl, model, loss_fn, optimizer,  alpha)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

#####################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion, print_freq):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, _, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg

def warmup_lr(epoch, step, optimizer, one_epoch_step, warmup, lr0):
    overall_steps = warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = lr0 * current_steps / overall_steps
    lr = min(lr, lr0)

    for p in optimizer.param_groups:
        p["lr"] = lr

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


import time
def FT_prune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    ** kwargs,

):
    ##################################### Training setting #################################################
    # lr = 0.1  # "initial learning rate"
    # warmup = 0  # "warm up epochs"
    # decreasing_lr = "91,136"  # "decreasing strategy"
    # decreasing_lr = list(map(int, decreasing_lr.split(",")))
    momentum = 0.9  # "momentum"
    weight_decay = 5e-4  # "weight decay"
    print_freq = 200  # "print frequency"
    ##################################### Unlearn setting #################################################
    with_l1 = True
    no_l1_epochs = 0  # "non l1 epochs"
    unlearn_lr = 0.01  # "initial learning rate"
    unlearn_epochs = 10  # "number of total epochs for unlearn to run"
    alpha = 5e-5  # "unlearn noise"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed

    for epoch in range(0, unlearn_epochs):
        start_time = time.time()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()

        for i, (image, _, target) in enumerate(retain_train_dl):
            # if epoch < warmup:
            #     warmup_lr(epoch, i + 1, optimizer, len(retain_train_dataloader), warmup, lr)

            image = image.cuda()
            target = target.cuda()
            if epoch < unlearn_epochs - no_l1_epochs:
                current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))  # decaying
                ## current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))  # increasing
            elif unlearn_epochs - no_l1_epochs == 0:
                current_alpha = alpha
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss += current_alpha * l1_regularization(model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % print_freq == 0:
                end = time.time()
                print("Epoch: [{0}][{1}/{2}]\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                      "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Time {3:.2f}".format(epoch, i, len(retain_train_dl), end - start, loss=losses, top1=top1))
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

        # scheduler.step()

        print("one epoch duration:{}".format(time.time() - start_time))

    # val
    validate(valid_dl, model, criterion, print_freq)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

########
from torch.autograd import grad
def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)

def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr: curr + length].view(param.shape)
            curr += length

def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def woodfisher_im(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 300000
    o_vec = None
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    for idx, batch in enumerate(tqdm(train_dl)):
        data, labels, clabels = batch
        model.zero_grad()
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

def Woodfisher(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        **kwargs,
):
    alpha = 1
    batch_size = 32
    criterion = nn.CrossEntropyLoss()

    retain_grad_loader = torch.utils.data.DataLoader(
        retain_train_dl.dataset, batch_size=batch_size, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_train_dl.dataset, batch_size=1, shuffle=False
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_train_dl.dataset, batch_size=batch_size, shuffle=False
    )

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()

    for i, batch in enumerate(tqdm(forget_loader)):
        data, labels, clabels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, batch in enumerate(tqdm(retain_grad_loader)):
        data, labels, clabels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = clabels.to(device)
        output = model(data)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss) * real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad,
    )
    apply_perturb(model, alpha * perturb)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )
