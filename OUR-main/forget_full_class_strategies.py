import random
from copy import deepcopy

from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
from collections import OrderedDict

import models
from repdistiller.distiller_zoo import DistillKL
from repdistiller.helper.loops import validate_scrub, train_distill
from repdistiller.helper.util import adjust_learning_rate
from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import config
import time

# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[clabel].append((img, label, clabel))
    return classwise_ds

# Returns metrics
def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    valid_dl,
    device,
    fast=True
):
    # loss_acc_dict = evaluate(model, valid_dl, device)
    # retain_acc_dict = evaluate(model, retain_valid_dl, device)
    # zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    # d_f = evaluate(model, forget_valid_dl, device)
    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    loss_acc_dict = evaluate(model, retain_valid_dl, device)
    d_f_acc_dict = evaluate(model, forget_train_dl, device)
    retain_acc_dict = evaluate(model, retain_train_dl, device)
    # zrf = UnLearningScore(model, unlearning_teacher, forget_train_dl, 128, device)
    if fast:
        mia = 0.0
    else:
        mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)

    return loss_acc_dict["Acc"], d_f_acc_dict["Acc"], retain_acc_dict["Acc"], mia

# Does nothing; original model
def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
        weights_path,
    **kwargs,
):
    end = time.time()
    start = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed


# Retrain the model on the retrain dataset only
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
    num_classes,
    weights_path,
    para1='0.0003',
    para2 = '150',
    **kwargs,
):
    start = time.time()
    model = getattr(models, model_name)(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.cuda()

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
        lr=float(para1),
        milestones=milestones,
        device=device,
        model_name=model_name,
    )
    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed


# Finetune the model using the retain data for a set number of epochs
# def finetune(
#     model,
#     unlearning_teacher,
#     retain_train_dl,
#     retain_valid_dl,
#     forget_train_dl,
#     forget_valid_dl,
#     valid_dl,
#     device,
#     mask=None,
#     weights_path=None,
#     **kwargs,
# ):
#     start = time.time()
#     _ = fit_one_cycle(
#         10, model, retain_train_dl, retain_valid_dl, lr=0.01, device=device, mask=mask
#     )
#     end = time.time()
#     time_elapsed = end - start
#     torch.save(model.state_dict(), weights_path)
#     return get_metric_scores(
#         model,
#         unlearning_teacher,
#         retain_train_dl,
#         retain_valid_dl,
#         forget_train_dl,
#         valid_dl,
#         device,
#     ), time_elapsed

def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    weights_path,
    para1='0.1',
    para2 = '10',
        rum=False,
        model_name='ResNet18',
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
    with_l1 = False
    no_l1_epochs = 0  # "non l1 epochs"
    unlearn_lr = float(para1)  # "initial learning rate"
    unlearn_epochs = int(para2)  # "number of total epochs for unlearn to run"
    alpha = 5e-5  # "unlearn noise"

    criterion = nn.CrossEntropyLoss()
    if model_name == "ViT":
        optimizer = optim.AdamW(model.parameters(), lr=unlearn_lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed
    start = time.time()

    for epoch in range(0, unlearn_epochs):
        start_time = time.time()
        print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))

        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()

        for i, (image, _, target) in enumerate(retain_train_dl):
            # if epoch < warmup:
            #     warmup_lr(epoch, i + 1, optimizer, len(retain_train_dl), warmup, lr)

            image = image.cuda()
            target = target.cuda()
            if epoch < unlearn_epochs - no_l1_epochs:
                current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))  # decaying
                ## current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))  # increasing
            elif unlearn_epochs  - no_l1_epochs == 0:
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
                print("Epoch: [{0}][{1}/{2}]\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                      "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                      "Time {3:.2f}".format(epoch, i, len(retain_train_dl), time.time() - start_time,
                                            loss=losses, top1=top1))

        # scheduler.step()

    end = time.time()
    time_elapsed = end - start
    # val
    validate(valid_dl, model, criterion, print_freq)

    torch.save(model.state_dict(), weights_path)

    if rum:
        return model
    else:
        return get_metric_scores(
            model,
            unlearning_teacher,
            retain_train_dl,
            retain_valid_dl,
            forget_train_dl,
            valid_dl,
            device,
        ), time_elapsed

# Implementation from https://github.com/vikram2000b/bad-teaching-unlearning
def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    mask=None,
    weights_path=None,
    para1=0.0001,
    para2=3,
    **kwargs,
):
    start = time.time()
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    unlearninglabels.remove(forget_class)

    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    for x, _, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, _, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        int(para2), model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=float(para1), mask=mask
    )

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    ), time_elapsed

# Extremely slow >>> Fisher https://github.com/AdityaGolatkar/SelectiveForgetting
def NTK(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
        weights_path,
    **kwargs,
):
    def delta_w_utils(model_init, dataloader, name="complete"):
        model_init.eval()
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False
        )
        G_list = []
        f0_minus_y = []
        for idx, batch in enumerate(
            tqdm(dataloader)
        ):  # (tqdm(dataloader,leave=False)):
            batch = [
                tensor.to(next(model_init.parameters()).device) for tensor in batch
            ]
            input, _, target = batch

            target = target.cpu().detach().numpy()
            output = model_init(input)
            G_sample = []
            for cls in range(num_classes):
                grads = torch.autograd.grad(
                    output[0, cls], model_init.parameters(), retain_graph=True
                )
                grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
                G_sample.append(grads)
                G_list.append(grads)
                p = (
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                    .transpose()
                )
                p[target] -= 1
                f0_y_update = deepcopy(p)
            f0_minus_y.append(f0_y_update)
        return np.stack(G_list).transpose(), np.vstack(f0_minus_y)

    #############################################################################################
    start = time.time()
    model_init = deepcopy(model)
    G_r, f0_minus_y_r = delta_w_utils(deepcopy(model), retain_train_dl, "complete")
    print("GOT GR")
    # np.save('NTK_data/G_r.npy',G_r)
    # np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
    # del G_r, f0_minus_y_r

    G_f, f0_minus_y_f = delta_w_utils(deepcopy(model), forget_train_dl, "retain")
    print("GOT GF")
    # np.save('NTK_data/G_f.npy',G_f)
    # np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
    # del G_f, f0_minus_y_f

    # G_r = np.load('NTK_data/G_r.npy')
    # G_f = np.load('NTK_data/G_f.npy')
    G = np.concatenate([G_r, G_f], axis=1)
    print("GOT G")
    # np.save('NTK_data/G.npy',G)
    # del G, G_f, G_r

    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    # f0_minus_y_f = np.load('NTK_data/f0_minus_y_f.npy')
    f0_minus_y = np.concatenate([f0_minus_y_r, f0_minus_y_f])

    # np.save('NTK_data/f0_minus_y.npy',f0_minus_y)
    # del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    weight_decay = 0.1

    # G = np.load('NTK_data/G.npy')
    theta = G.transpose().dot(G) + (
        len(retain_train_dl.dataset) + len(forget_train_dl.dataset)
    ) * weight_decay * np.eye(G.shape[1])
    # del G

    theta_inv = np.linalg.inv(theta)

    # np.save('NTK_data/theta.npy',theta)
    # del theta

    # G = np.load('NTK_data/G.npy')
    # f0_minus_y = np.load('NTK_data/f0_minus_y.npy')
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    # np.save('NTK_data/theta_inv.npy',theta_inv)
    # np.save('NTK_data/w_complete.npy',w_complete)
    # del G, f0_minus_y, theta_inv, w_complete

    # G_r = np.load('NTK_data/G_r.npy')
    num_to_retain = len(retain_train_dl.dataset)
    theta_r = G_r.transpose().dot(G_r) + num_to_retain * weight_decay * np.eye(
        G_r.shape[1]
    )
    # del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    # np.save('NTK_data/theta_r.npy',theta_r)
    # del theta_r

    # G_r = np.load('NTK_data/G_r.npy')
    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    # np.save('NTK_data/theta_r_inv.npy',theta_r_inv)
    # np.save('NTK_data/w_retain.npy',w_retain)
    # del G_r, f0_minus_y_r, theta_r_inv, w_retain

    def get_delta_w_dict(delta_w, model):
        # Give normalized delta_w
        delta_w_dict = OrderedDict()
        params_visited = 0
        for k, p in model.named_parameters():
            num_params = np.prod(list(p.shape))
            update_params = delta_w[params_visited : params_visited + num_params]
            delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
            params_visited += num_params
        return delta_w_dict

    #### Scrubbing Direction
    # w_complete = np.load('NTK_data/w_complete.npy')
    # w_retain = np.load('NTK_data/w_retain.npy')
    print("got prelims, calculating delta_w")
    delta_w = (w_retain - w_complete).squeeze()
    print("got delta_w")
    # delta_w_copy = deepcopy(delta_w)
    # delta_w_actual = vectorize_params(model0)-vectorize_params(model)

    # print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    # print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    # scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    # print('Actual Scale: {}'.format(scale_ratio))
    # log_dict['actual_scale_ratio']=scale_ratio
    def vectorize_params(model):
        param = []
        for p in model.parameters():
            param.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(param)

    m_pred_error = (
        vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()
    )
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(
        delta_w / np.linalg.norm(delta_w), m_pred_error / np.linalg.norm(m_pred_error)
    )
    print(f"Inner Product--: {inner}")

    if inner < 0:
        angle = np.arccos(inner) - np.pi / 2
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.sin(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner)
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.cos(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale = predicted_norm / np.linalg.norm(delta_w)
    predicted_scale
    print(f"Predicted Scale:  {predicted_scale}")
    # log_dict['predicted_scale_ratio']=predicted_scale

    # def NIP(v1,v2):
    #     nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    #     print(nip)
    #     return nip
    # nip=NIP(delta_w_actual,delta_w)
    # log_dict['nip']=nip
    scale = predicted_scale
    direction = get_delta_w_dict(delta_w, model)

    for k, p in model.named_parameters():
        p.data += (direction[k] * scale).to(device)

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed


# From https://github.com/AdityaGolatkar/SelectiveForgetting
def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    mask=None,
        para1='1e-8',
        para2='0',
        weights_path=None,
    **kwargs,
):
    load=False
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
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.size(0) == num_classes:
            mu[forget_class] = 0
            var[forget_class] = 0.0001
        if p.size(0) == num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    start = time.time()
    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = float(para1)
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed

# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    weights_path,
    **kwargs,
):

    start = time.time()
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), num_classes
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_classes):
        if i != forget_class:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_class
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 35, noise_batch_size, device=device
    )#25
    #high effective
    noisy_loader = UNSIR_create_noisy_loader(
        noise,
        forget_class_label,
        retain_samples,
        batch_size=noise_batch_size,
        device=device,
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.001
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

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed

#1
def ssd_tuning(
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
        weights_path,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": 0.5,#dampening_constant,  #1,  Lambda from paper
        "selection_weighting": 5#selection_weighting,  #10,  Alpha from paper
    }

    start = time.time()
    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    ), time_elapsed

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

def train_negrad_1(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, alpha,
                   mask=None, prune=False, epoch=0):
    # MAE = nn.L1Loss()
    model.train()
    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
    # for idx, (batch_forget) in enumerate(forget_loader):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r, _, target_r = batch_retain
        input_f, _, target_f = batch_forget
        output_r = model(input_r)
        output_f = model(input_f)
        loss = alpha * (loss_fn(output_r, target_r) +
                        l2_penalty(model, model_init, weight_decay=0.1)) - \
               (1 - alpha) * loss_fn(output_f, target_f)

        if prune:
            loss += (1e-3) * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        # if mask and epoch < 3:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param.grad *= (1.0 - mask[epoch][name])
        # if mask:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param.grad *= mask[name]

        optimizer.step()

    # with torch.no_grad():
    #     model.eval()
    #     print(f"Epoch[{epoch}]:", "Retain Dataset Acc", evaluate(model, retain_loader, next(model.parameters()).device)["Acc"])
    return

def train_negrad_2(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, alpha, mask=None, prune=False,
                   epoch=0, num_classes=19, forget_class=4):
    model.train()
    total_len = len(retain_loader)
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []
    unlearninglabels.remove(forget_class)
    unlearninglabels = torch.tensor(unlearninglabels)

    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r, _, target_r = batch_retain
        input_f, _, _ = batch_forget

        indices = torch.randint(low=0, high=len(unlearninglabels), size=(len(input_f),))
        target_f = torch.index_select(unlearninglabels, 0, indices).to(next(model.parameters()).device)

        output_r = model(input_r)
        output_f = model(input_f)
        loss = alpha * (loss_fn(output_r, target_r) +
                        l2_penalty(model, model_init, weight_decay=0.1)) - \
               (1 - alpha) * loss_fn(output_f, target_f)

        if prune:
            loss += (1e-3) * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    return

import copy
import torch.nn as nn

def ref_f(dataloader, Classifier, num_classes, device):
    Classifier.eval()
    F = {}
    F_out = []
    for fs, _, labels in (dataloader):
        batch_fs = fs
        break
    for ii in range(num_classes):
        F[ii] = [torch.zeros_like(Classifier(batch_fs[:1].to(device))[0, :]).to(device)]
    batch_size = len(batch_fs)
    # print("batch_size", batch_size)
    # print("F[ii]", F[ii][0].shape)
    for fs,_,labels in (dataloader):
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        features = Classifier(fs)
        for ii in (range(fs.shape[0])):
            label = labels[ii].item()
            F[label].append(features[ii, :])

    for ii in range(num_classes):
        F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(batch_size, dim_f)
        F_out.append(F[ii])
        # print("F[ii]", F[ii].shape)

    F_out = torch.stack(F_out).mean(dim=0)
    # print("F_out", F_out.shape)
    F_out = F_out.reshape(-1, dim_f).detach().cpu()#[num_classes, batch_size, num_classes]
    return F_out.cuda() #it's a? how to say?how to determine? every dim?

#2
def negative_grad(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    mask=None,
        weights_path=None,
        para1='0.003',
        para2='5',
        model_name='ResNet18',
    **kwargs,
):
    alpha = 0.95
    #the defending para
    # epochs = 3#5
    # lr = 0.01#0.0001#0.01
    #the baseline para
    epochs = int(para2)
    lr = float(para1)#0.01
    quiet = True
    loss_fn = nn.CrossEntropyLoss()
    if model_name == 'ViT':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)#always the Adam?

    #useless
    model_init = copy.deepcopy(model)
    # num_classes = 19
    # feature_ini = ref_f(forget_train_dl, model_init, num_classes, device)
    start = time.time()
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_1(model, model_init, retain_train_dl, forget_train_dl, loss_fn, optimizer, alpha,
                       mask, epoch=epoch)

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True,
    ), time_elapsed

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

#1
def FT_prune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
        weights_path,
        para1='0.01',
        para2='10',
        model_name='ResNet18',
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
    unlearn_lr = float(para1)  # "initial learning rate"
    unlearn_epochs = int(para2)  # "number of total epochs for unlearn to run"
    alpha = 5e-5  # "unlearn noise"

    criterion = nn.CrossEntropyLoss()
    if model_name == "ViT":
        optimizer = optim.AdamW(model.parameters(), lr=unlearn_lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), unlearn_lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed
    start = time.time()

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
            #     warmup_lr(epoch, i + 1, optimizer, len(retain_train_dl), warmup, lr)

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

    end = time.time()
    time_elapsed = end - start
    # val
    validate(valid_dl, model, criterion, print_freq)

    torch.save(model.state_dict(), weights_path)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
    ), time_elapsed

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

#4
def Wfisher(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        weights_path=None,
        para1='3',
        para2='0',
        **kwargs,
):
    alpha = float(para1)
    print("alpha:", alpha)

    device = next(model.parameters()).device
    print("device:", device)

    batch_size = 64
    criterion = nn.CrossEntropyLoss()

    start = time.time()

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

    # _ = fit_one_cycle(
    #     5, model, retain_train_dl, retain_valid_dl, lr=0.01, device=next(model.parameters()).device
    # )

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True,
    ), time_elapsed

import copy
import boundary_utils
from boundary_unlearning.trainer import eval, loss_picker, optimizer_picker
import torch
from torch import nn, optim
from boundary_unlearning.adv_generator import inf_generator, FGSM
import time
from boundary_unlearning.boundary_utils import init_params as w_init
from expand_exp import curvature, weight_assign

#boundary type
def boundary_shrink(model,
                    unlearning_teacher,
                    retain_train_dl,
                    retain_valid_dl,
                    forget_train_dl,
                    forget_valid_dl,
                    valid_dl,
                    device,
                    weights_path,
                    para1='0.1',
                    para2='10',
                    extra_exp=None, lambda_=0.7,
                    bias=-0.5, slope=5.0, **kwargs):
    bound = float(para1)
    poison_epoch = int(para2)
    start = time.time()
    norm = True  # None#True if data_name != "mnist" else False
    random_start = False  # False if attack != "pgd" else True

    test_model = copy.deepcopy(model).to(device)
    ori_model = copy.deepcopy(model).to(device)

    # adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)
    adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = inf_generator(forget_train_dl)
    batches_per_epoch = len(forget_train_dl)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)#TODO 0.00001

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm(range(poison_epoch * batches_per_epoch)):
        x, _, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        ori_logits = model(x)
        ori_loss = criterion(ori_logits, pred_label)

        # loss = ori_loss  # - KL_div
        if extra_exp == 'curv':
            ori_curv = curvature(ori_model, x, y, h=0.9)[1]
            cur_curv = curvature(model, x, y, h=0.9)[1]
            delta_curv = torch.norm(ori_curv - cur_curv, p=2)
            loss = ori_loss + lambda_ * delta_curv  # - KL_div
        elif extra_exp == 'weight_assign':
            weight = weight_assign(adv_logits, pred_label, bias=bias, slope=slope)
            ori_loss = (torch.nn.functional.cross_entropy(ori_logits, pred_label, reduction='none') * weight).mean()
            loss = ori_loss
        else:
            loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    # print(nearest_label)
    print('boundary shrink time:', (time.time() - start))
    end = time.time()
    time_elapsed = end - start
    # np.save('nearest_label', nearest_label)
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,), time_elapsed

def boundary_expanding(model,
                       unlearning_teacher,
                       retain_train_dl,
                       retain_valid_dl,
                       forget_train_dl,
                       forget_valid_dl,
                       valid_dl,
                       num_classes,
                       weights_path,
                       device, **kwargs):

    start=time.time()
    n_filter2 = int(192 * 0.5)
    num_classes = int(num_classes)
    optimization = 'sgd'

    widen_fc = nn.Linear(n_filter2, num_classes + 1).to(device)
    fc = model.fc
    w_init(widen_fc)
    widen_model = copy.deepcopy(model)
    widen_model.fc = widen_fc
    widen_model = widen_model.to(device)

    for name, params in fc.named_parameters():
        # print(name, params.data)
        if 'weight' in name:
            fc.state_dict()['weight'][0:10,] = fc.state_dict()[name][:, ]
        elif 'bias' in name:
            fc.state_dict()['bias'][0:10,] = fc.state_dict()[name][:, ]

    forget_data_gen = inf_generator(forget_train_dl)
    batches_per_epoch = len(forget_train_dl)
    finetune_epochs = 10

    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)

    for itr in tqdm(range(finetune_epochs * batches_per_epoch)):
        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)

        widen_logits = widen_model(x)
        # target label
        target_label = torch.ones_like(y, device=device)
        target_label *= num_classes

        # adv_train
        widen_model.train()
        widen_model.zero_grad()
        optimizer.zero_grad()

        widen_loss = criterion(widen_logits,
                               target_label)

        widen_loss.backward()
        optimizer.step()

    pruned_classifier = nn.Linear(n_filter2, num_classes)
    for name, params in widen_model[1].named_parameters():
        # print(name)
        if 'weight' in name:
            pruned_classifier.state_dict()['weight'][:, ] = widen_model[1].state_dict()[name][0:10, ]
        elif 'bias' in name:
            pruned_classifier.state_dict()['bias'][:, ] = widen_model[1].state_dict()[name][0:10, ]

    pruned_model = copy.deepcopy(model)
    pruned_model.fc = pruned_classifier
    model = pruned_model.to(device)
    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,), time_elapsed


def split_forget_dataset_by_mem(forget_dataset, forget_memorization, part_num=3):
    """
    将 forget_dataset 中的每个样本根据其 mem 值（第三个返回值）
    从低到高排序后，均分为三个子数据集：
      - low_forget_dataset：mem 较低的样本
      - mid_forget_dataset：mem 中间的样本
      - high_forget_dataset：mem 较高的样本
    返回值为 (low_forget_dataset, mid_forget_dataset, high_forget_dataset)
    """
    all_indices = list(range(len(forget_dataset)))
    mem_values = forget_memorization

    # 2. 对所有索引按照 mem 值从低到高排序
    sorted_indices = [idx for _, idx in sorted(zip(mem_values, all_indices), key=lambda pair: pair[0])]

    # 3. 均分索引列表
    n = len(sorted_indices)
    # 计算每一份的大小（余数部分归入最后一份）
    if part_num == 3:
        third = n // 3
        # 为了确保三份包含所有样本，可以这样划分：
        low_indices = sorted_indices[:third]
        mid_indices = sorted_indices[third:2 * third]
        high_indices = sorted_indices[2 * third:]

        # 4. 利用 Subset 构造三个子数据集
        low_forget_dataset = Subset(forget_dataset, low_indices)
        mid_forget_dataset = Subset(forget_dataset, mid_indices)
        high_forget_dataset = Subset(forget_dataset, high_indices)

        return low_forget_dataset, mid_forget_dataset, high_forget_dataset
    elif part_num == 2:
        split_num = int(n *0.7)
        # 为了确保三份包含所有样本，可以这样划分：
        low_indices = sorted_indices[:split_num]
        high_indices = sorted_indices[split_num:]

        # 4. 利用 Subset 构造三个子数据集
        low_forget_dataset = Subset(forget_dataset, low_indices)
        high_forget_dataset = Subset(forget_dataset, high_indices)

        return low_forget_dataset, high_forget_dataset

def salun(model,
          unlearning_teacher,
          retain_train_dl,
          retain_valid_dl,
          forget_train_dl,
          valid_dl,  num_classes, device, weights_path, para1='0.0001', para2='2',
             mask_path=None, rum=False,
             **kwargs):
    start_time=time.time()
    # TODO salun mask
    mask = torch.load(mask_path)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(para1))

    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dl) + len(retain_train_dl)

        for i, batch in enumerate(forget_train_dl):
            # if rum:
            #     image, target, _ = batch
            # else:
            #     image, _, target = batch
            image, _, target = batch
            image = image.cuda()
            target = torch.randint(0, num_classes, target.shape).cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters(): #TODO .module.
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

        for i, batch in enumerate(retain_train_dl):
            # if rum:
            #     image, target, _ = batch
            # else:
            #     image, _, target = batch
            image, _, target = batch

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters(): #TODO .module.
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

        if (i + 1) % 100 == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                epoch, i, loader_len, end - start, loss=losses, top1=top1))
            start = time.time()
    time_elapsed = time.time() - start_time
    if not rum:
        torch.save(model.state_dict(), weights_path)
        return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl,
                             forget_train_dl, valid_dl,  device, fast=True), time_elapsed
    else:
        return model

def rum(model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
            device, weights_path, forget_dataset, forget_memorization, num_classes, para1=0.1, para2=0.00005, mask_path=None, forget_dataset_memorization=None,
            **kwargs):
    start_time = time.time()
    # meta unlearn
    # nothing-Finetune-SalUn (low-medium-high memorization order)
    # Step1: according to the memorization to seperate the dataloader into three subsets
    low_forget_dataset, mid_forget_dataset, high_forget_dataset = split_forget_dataset_by_mem(forget_dataset, forget_memorization)

    # 下面可以对 low_forget_dataset、mid_forget_dataset、high_forget_dataset 分别进行后续操作
    # print("low_forget_dataset size:", len(low_forget_dataset))
    # print("mid_forget_dataset size:", len(mid_forget_dataset))
    # print("high_forget_dataset size:", len(high_forget_dataset))

    #TODO notice the structure of the batch

    # Step2: do noting to the low; do fientune to the medium forget dataset; do salun to the high forget dataset
    mid_forget_dataloader = DataLoader(mid_forget_dataset,128, shuffle=True)
    model = finetune(model=model,
                     unlearning_teacher=unlearning_teacher,
                     retain_train_dl=retain_train_dl,
                     retain_valid_dl=retain_valid_dl,
                     forget_train_dl=mid_forget_dataloader,
                     valid_dl=valid_dl,
                     device=device,
                     weights_path=weights_path, para1=para1, para2=5, mask_path=None, rum=True,
            **kwargs)

    high_forget_dataloader = DataLoader(high_forget_dataset,128, shuffle=True) #high_forget_dataset
    model = salun(model=model,
                    unlearning_teacher=unlearning_teacher,
                    retain_train_dl=retain_train_dl,
                    retain_valid_dl=retain_valid_dl,
                     forget_train_dl=mid_forget_dataloader,
                     valid_dl=valid_dl,
                     device=device,
                     weights_path=weights_path,
            num_classes=num_classes, para1=str(para2),
                  para2=3, mask_path=mask_path,  rum=True,
            **kwargs)

    time_elapsed = time.time() - start_time
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl,
                             forget_train_dl, valid_dl,  device, fast=True), time_elapsed

def sfron(model,
          unlearning_teacher,
          retain_train_dl,
          retain_valid_dl,
          forget_train_dl,
          forget_valid_dl,
          valid_dl,
          num_classes, 
          device, 
          weights_path, 
          para1='0.0001', 
          para2='2',
          mask_path=None,
          model_name='ResNet18',
             **kwargs):
    start_time = time.time()

    forget_freq = 5
    # TODO salun mask
    mask = torch.load(mask_path)
    criterion = torch.nn.CrossEntropyLoss()

    if model_name == 'ViT':
        optimizer = optim.AdamW(model.parameters(), lr=float(para1), weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=float(para1))
    for epoch in range(int(para2)):
        model.train()
        start = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        loader_len = len(forget_train_dl) + len(retain_train_dl)
        iter_forget_loader = iter(forget_train_dl)
        iter_retain_loader = iter(retain_train_dl)

        for step in range(len(retain_train_dl)):
            if step % forget_freq == 0:
                try:
                    image, _, target = next(iter_forget_loader)
                except:
                    iter_forget_loader = iter(forget_train_dl)
                    image, _, target = next(iter_forget_loader)

                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = model(image)
                loss = -criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                if mask:
                    for name, param in model.named_parameters(): #module
                        if param.grad is not None:
                            param.grad *= mask[name]

                optimizer.step()

            try:
                image, _, target = next(iter_retain_loader)
            except:
                iter_retain_loader = iter(retain_train_dl)
                image, _, target = next(iter_retain_loader)

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters(): #module
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (step + 1) % 100 == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, step, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()
    time_elapsed = time.time() - start_time
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl,
                             forget_train_dl, valid_dl, device, fast=True), time_elapsed

class Args:
    pass

def scrub(model,
          unlearning_teacher,
          retain_train_dl,
          retain_valid_dl,
          forget_train_dl,
          valid_dl,
            device, weights_path, para1=0.001, para2=5,
            **kwargs):
    model_t = copy.deepcopy(model)

    start_time = time.time()

    args = Args()
    args.optim = 'adam'#'sgd'
    args.gamma = 1
    args.alpha = 0.1
    args.beta = 0
    args.smoothing = 0.5
    args.msteps = 2 #first 5 epochs, maximize the training acc of forget dataset; after that, minimize
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = 'kd'
    args.print_freq = 50

    args.sgda_epochs = int(para2)
    args.sgda_learning_rate = float(para1)#0.0005
    args.lr_decay_epochs = [3, 5, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    model_s = copy.deepcopy(model)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args.sgda_learning_rate,
                              momentum=args.sgda_momentum,
                              weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=args.sgda_learning_rate,
                               weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                                  lr=args.sgda_learning_rate,
                                  momentum=args.sgda_momentum,
                                  weight_decay=args.sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    t1 = time.time()
    acc_rs = []
    acc_fs = []
    acc_vs = []
    acc_fvs = []

    forget_validation_loader = copy.deepcopy(valid_dl)
    # fgt_cls = list(np.unique(forget_train_dl.dataset.targets))
    # indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]
    # forget_validation_loader.dataset.data = forget_validation_loader.dataset.data[indices]
    # forget_validation_loader.dataset.targets = forget_validation_loader.dataset.targets[indices]

    # scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(args.model, args.dataset, args.seed)
    for epoch in range(1, args.sgda_epochs + 1):

        lr = adjust_learning_rate(epoch, args, optimizer)

        acc_r, acc5_r, loss_r = validate_scrub(retain_train_dl, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate_scrub(forget_train_dl, model_s, criterion_cls, args, True)
        acc_v, acc5_v, loss_v = validate_scrub(valid_dl, model_s, criterion_cls, args, True)
        acc_fv, acc5_fv, loss_fv = validate_scrub(forget_validation_loader, model_s, criterion_cls, args, True)
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())
        acc_vs.append(100 - acc_v.item())
        acc_fvs.append(100 - acc_fv.item())

        maximize_loss = 0
        if epoch <= args.msteps: #first three epochs
            maximize_loss = train_distill(epoch, forget_train_dl, module_list, None, criterion_list, optimizer, args,
                                          "maximize")
        #last two epoch
        train_acc, train_loss = train_distill(epoch, retain_train_dl, module_list, None, criterion_list, optimizer, args,
                                              "minimize")
        # torch.save(model_s.state_dict(), scrub_name + str(epoch) + ".pt")
        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss,
                                                                                     train_acc))
    t2 = time.time()
    print(t2 - t1)

    acc_r, acc5_r, loss_r = validate_scrub(retain_train_dl, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate_scrub(forget_train_dl, model_s, criterion_cls, args, True)
    acc_v, acc5_v, loss_v = validate_scrub(valid_dl, model_s, criterion_cls, args, True)
    acc_fv, acc5_fv, loss_fv = validate_scrub(forget_validation_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100 - acc_r.item())
    acc_fs.append(100 - acc_f.item())
    acc_vs.append(100 - acc_v.item())
    acc_fvs.append(100 - acc_fv.item())

    from matplotlib import pyplot as plt
    # indices = list(range(0, len(acc_rs)))
    # plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    # plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    # plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c', alpha=1, label='validation-set')
    # plt.plot(indices, acc_fvs, marker='.', color='red', alpha=1, label='forget-validation-set')
    # plt.legend(prop={'size': 14})
    # plt.tick_params(labelsize=12)
    # plt.xlabel('epoch', size=14)
    # plt.ylabel('error', size=14)
    # plt.grid()
    # plt.show()

    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1] - acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1
    print("the selected index is {}".format(selected_idx))
    # selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(args.model, args.dataset, args.seed,
    #                                                                    int(selected_idx))
    model = copy.deepcopy(model_s)
    # model_s.load_state_dict(torch.load(selected_model))
    # return model_s, model_s_final
    time_elapsed = time.time() - start_time
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(model, unlearning_teacher, retain_train_dl, retain_valid_dl,
                             forget_train_dl, valid_dl,  device, fast=True), time_elapsed



##############################
#Ours
def orthogonality(model,
                    unlearning_teacher,
                    retain_train_dl,
                    retain_valid_dl,
                    forget_train_dl,
                    forget_valid_dl,
                    valid_dl,
                    num_classes,
                    forget_class,
                    device,
                    weights_path,
                    para1 ='0.0015',
                    para2 ='0.01',
                  model_name = 'ResNet18',
                    **kwargs):
    start = time.time()

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []
    unlearninglabels.remove(forget_class)
    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    # for idx, (x, _, y) in enumerate(retain_train_dl.dataset):
    #     unlearning_trainset.append((x, _, y))
    #     if idx > 5000:
    #         break

    unlearned_train_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    ortho_gamma = 2.0
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            batch_size=output.size(0)
            activation[name] = output.view(batch_size, -1)
        return hook

    def orthogonality_loss(features):
        """
        计算正交性损失：
          - 输入 features 形状为 (N, d)（每个样本被 flatten 成 d 维向量）
          - 归一化每个样本向量，然后计算 Gram 矩阵；
          - 对非对角项施加平方惩罚。
        """
        eps = 1e-8
        norm = features.norm(dim=1, keepdim=True) + eps
        features_norm = features / norm
        gram = torch.mm(features_norm, features_norm.t())
        eye = torch.eye(gram.size(0), device=gram.device)
        loss = ((gram - eye) ** 2).sum() / (gram.size(0) * (gram.size(0) - 1) + eps)
        return loss

    def fit_one_unlearning_cycle_orthogonal(epochs, model, train_loader, lr, device, mask=None, model_name='ResNet18'):
        history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                images, _, labels = batch
                images, labels = images.to(device), labels.to(device)
                out = model(images)  # Generate predictions

                # batch_features0 = activation.get('features0', None)
                batch_features1 = activation.get('features1', None)
                batch_features2 = activation.get('features2', None)
                batch_features3 = activation.get('features2', None)

                loss_ortho = 0.0
                if batch_features1 is not None:
                    loss_ortho = (
                                  orthogonality_loss(batch_features1.to(device)) +
                                  orthogonality_loss(batch_features2.to(device)) +
                                  orthogonality_loss(batch_features3.to(device))) #orthogonality_loss(batch_features0.to(device)) +

                loss = ortho_gamma * loss_ortho + F.cross_entropy(out, labels)

                loss.backward()
                train_losses.append(loss.detach().cpu())

                # if mask:
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             param.grad *= mask[name]

                optimizer.step()
                optimizer.zero_grad()

                lrs.append(get_lr(optimizer))

            result = evaluate(model, valid_dl, device)
            result["train_loss"] = torch.stack(train_losses).mean()
            result["lrs"] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
        return history

    # 在 feature_extractor 中的 ReLU 层注册 forward hook，捕获激活值
    #for resnet18
    # hook_handle0 = model.module.conv2_x[-1].register_forward_hook(get_activation('features0'))
    # print(model.module.transformer)

    if model_name == 'ResNet18':
        hook_handle1 = model.module.conv3_x[-1].register_forward_hook(get_activation('features1'))
        hook_handle2 = model.module.conv4_x[-1].register_forward_hook(get_activation('features2'))
        hook_handle3 = model.module.conv5_x[-1].register_forward_hook(get_activation('features3'))
    elif model_name == 'ViT':
        hook_handle1 = model.module.transformer.layers[0][1].fn.net[0].register_forward_hook(get_activation('features1'))
        hook_handle2 = model.module.transformer.layers[3][0].fn.to_qkv.register_forward_hook(get_activation('features2'))
        hook_handle3 = model.module.mlp_head[0].register_forward_hook(get_activation('features3'))

    #unlearn process of the amnesiac
    _ = fit_one_unlearning_cycle_orthogonal(3, model, unlearned_train_dl,
                                            device=device, lr=float(para1), model_name=model_name)
    torch.save(model.state_dict(), weights_path[:-4]+'_mid.pth')
    print("Will save model to ", weights_path[:-4]+'_mid.pth')
    d_t, d_f, d_r, mia =get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    )
    print("d_t = ", d_t, "| d_f = ", d_f, "| d_r = ", d_r, "| mia = ", mia)

    _ = fit_one_cycle(
        3, model, retain_train_dl, retain_valid_dl, lr=float(para2),
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )
    _ = fit_one_cycle(
        2, model, retain_train_dl, retain_valid_dl, lr=float(para2)*0.5,
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )
    _ = fit_one_cycle(
        1, model, retain_train_dl, retain_valid_dl, lr=float(para2)*0.2,
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )

    # hook_handle0.remove()
    hook_handle1.remove()
    hook_handle2.remove()
    hook_handle3.remove()

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    ), time_elapsed

def orth_no_our(model,
                    unlearning_teacher,
                    retain_train_dl,
                    retain_valid_dl,
                    forget_train_dl,
                    forget_valid_dl,
                    valid_dl,
                    num_classes,
                    forget_class,
                    device,
                    weights_path,
                    para1 ='0.0015',
                    para2 = '3',
                  model_name = 'ResNet18',
                    **kwargs):
    start = time.time()

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []
    unlearninglabels.remove(forget_class)
    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    # for idx, (x, _, y) in enumerate(retain_train_dl.dataset):
    #     unlearning_trainset.append((x, _, y))
    #     if idx > 5000:
    #         break

    unlearned_train_dl = DataLoader(
        unlearning_trainset, 64, pin_memory=True, shuffle=True
    )

    ortho_gamma = 2.0
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            batch_size=output.size(0)
            activation[name] = output.view(batch_size, -1)
        return hook

    def orthogonality_loss(features):
        """
        计算正交性损失：
          - 输入 features 形状为 (N, d)（每个样本被 flatten 成 d 维向量）
          - 归一化每个样本向量，然后计算 Gram 矩阵；
          - 对非对角项施加平方惩罚。
        """
        eps = 1e-8
        norm = features.norm(dim=1, keepdim=True) + eps
        features_norm = features / norm
        gram = torch.mm(features_norm, features_norm.t())
        eye = torch.eye(gram.size(0), device=gram.device)
        loss = ((gram - eye) ** 2).sum() / (gram.size(0) * (gram.size(0) - 1) + eps)
        return loss

    def fit_one_unlearning_cycle_orthogonal(epochs, model, forget_loader, retain_loader, lr, device, mask=None, model_name='ResNet18'):
        history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            for (batch, retain_batch) in zip(forget_loader, retain_loader):
                images, _, labels = batch
                retain_images, _, retain_labels = retain_batch
                images, labels = images.to(device), labels.to(device)
                retain_images, retain_labels = retain_images.to(device), retain_labels.to(device)
                out = model(images)  # Generate predictions

                # batch_features0 = activation.get('features0', None)
                batch_features1 = activation.get('features1', None)
                batch_features2 = activation.get('features2', None)
                batch_features3 = activation.get('features2', None)

                loss_ortho = 0.0
                if batch_features1 is not None:
                    loss_ortho = (
                                  orthogonality_loss(batch_features1.to(device)) +
                                  orthogonality_loss(batch_features2.to(device)) +
                                  orthogonality_loss(batch_features3.to(device))) #orthogonality_loss(batch_features0.to(device)) +

                loss = 0.8*(ortho_gamma * loss_ortho + F.cross_entropy(out, labels) +
                        F.cross_entropy(model(retain_images), retain_labels) + 5e-5*l1_regularization(model))

                loss.backward()
                train_losses.append(loss.detach().cpu())

                # if mask:
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             param.grad *= mask[name]

                optimizer.step()
                optimizer.zero_grad()

                lrs.append(get_lr(optimizer))

            result = evaluate(model, valid_dl, device)
            result["train_loss"] = torch.stack(train_losses).mean()
            result["lrs"] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
        return history

    # 在 feature_extractor 中的 ReLU 层注册 forward hook，捕获激活值
    #for resnet18
    # hook_handle0 = model.module.conv2_x[-1].register_forward_hook(get_activation('features0'))
    # print(model.module.transformer)

    if model_name == 'ResNet18':
        hook_handle1 = model.module.conv3_x[-1].register_forward_hook(get_activation('features1'))
        hook_handle2 = model.module.conv4_x[-1].register_forward_hook(get_activation('features2'))
        hook_handle3 = model.module.conv5_x[-1].register_forward_hook(get_activation('features3'))
    elif model_name == 'ViT':
        hook_handle1 = model.module.transformer.layers[0][1].fn.net[0].register_forward_hook(get_activation('features1'))
        hook_handle2 = model.module.transformer.layers[3][0].fn.to_qkv.register_forward_hook(get_activation('features2'))
        hook_handle3 = model.module.mlp_head[0].register_forward_hook(get_activation('features3'))

    #unlearn process of the amnesiac
    _ = fit_one_unlearning_cycle_orthogonal(int(para2), model, unlearned_train_dl, retain_train_dl, device=device, lr=float(para1), model_name=model_name)

    d_t, d_f, d_r, mia =get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    )
    print("d_t = ", d_t, "| d_f = ", d_f, "| d_r = ", d_r, "| mia = ", mia)

    # hook_handle0.remove()
    hook_handle1.remove()
    hook_handle2.remove()
    hook_handle3.remove()

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    ), time_elapsed

def rl_our(model,
                    unlearning_teacher,
                    retain_train_dl,
                    retain_valid_dl,
                    forget_train_dl,
                    forget_valid_dl,
                    valid_dl,
                    num_classes,
                    forget_class,
                    device,
                    weights_path,
                    para1 ='0.0015',
                    para2 ='0.01',
                  model_name = 'ResNet18',
                    **kwargs):
    start = time.time()

    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []
    unlearninglabels.remove(forget_class)
    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    # for idx, (x, _, y) in enumerate(retain_train_dl.dataset):
    #     unlearning_trainset.append((x, _, y))
    #     if idx > 5000:
    #         break

    unlearned_train_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    def fit_one_unlearning_cycle_relabel(epochs, model, train_loader, lr, device, mask=None, model_name='ResNet18'):
        history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                images, _, labels = batch
                images, labels = images.to(device), labels.to(device)
                out = model(images)  # Generate predictions
                loss = F.cross_entropy(out, labels)
                loss.backward()
                train_losses.append(loss.detach().cpu())

                optimizer.step()
                optimizer.zero_grad()

                lrs.append(get_lr(optimizer))

            result = evaluate(model, valid_dl, device)
            result["train_loss"] = torch.stack(train_losses).mean()
            result["lrs"] = lrs
            epoch_end(model, epoch, result)
            history.append(result)
        return history

    _ = fit_one_unlearning_cycle_relabel(3, model, unlearned_train_dl, device=device, lr=float(para1), model_name=model_name)

    d_t, d_f, d_r, mia =get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    )
    print("d_t = ", d_t, "| d_f = ", d_f, "| d_r = ", d_r, "| mia = ", mia)

    _ = fit_one_cycle(
        3, model, retain_train_dl, retain_valid_dl, lr=float(para2),
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )
    _ = fit_one_cycle(
        2, model, retain_train_dl, retain_valid_dl, lr=float(para2)*0.5,
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )
    _ = fit_one_cycle(
        1, model, retain_train_dl, retain_valid_dl, lr=float(para2)*0.2,
        device=next(model.parameters()).device, model_name=model_name, l1=True
    )

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        valid_dl,
        device,
        fast=True
    ), time_elapsed
