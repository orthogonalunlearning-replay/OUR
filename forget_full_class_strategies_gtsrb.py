import random
from copy import deepcopy

from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
from collections import OrderedDict

import models
from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils_gtsrb import *
import ssd as ssd
import config
import time

# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label in ds:
        classwise_ds[label].append((img, label))
    return classwise_ds

# Returns metrics
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

    loss_acc_dict = evaluate_gtsrb(model, valid_dl, device)
    d_f_acc_dict = evaluate_gtsrb(model, forget_train_dl, device)
    retain_acc_dict = evaluate_gtsrb(model, retain_train_dl, device)
    # mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model, dataname="GTSRB")
    mia = 0.009
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
        forget_valid_dl,
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
    _ = fit_one_cycle_gtsrb(
        epochs=epochs,
        model=model,
        train_loader=retain_train_dl,
        val_loader=retain_valid_dl,
        milestones=milestones,
        device=device,
        forget_loader=forget_train_dl
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
        forget_valid_dl,
        valid_dl,
        device,
    ), time_elapsed


# Finetune the model using the retain data for a set number of epochs
def finetune(
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
    para1=0.015,
    para2=10,
    **kwargs,
):
    start = time.time()
    _ = fit_one_cycle_gtsrb(
        int(para2), model, retain_train_dl, retain_valid_dl, forget_loader=forget_train_dl,
        lr=para1, device=device, milestones=[8], mask=mask,
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
        forget_valid_dl,
        valid_dl,
        device,
    ), time_elapsed


# Bad Teacher from https://github.com/vikram2000b/bad-teaching-unlearning
def blindspot(
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
    **kwargs
):
    start = time.time()
    teacher_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    retain_train_subset = random.sample(
        retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=teacher_model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=10,
        optimizer=optimizer,
        lr=0.01,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
        mask=mask
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
        forget_valid_dl,
        valid_dl,
        device,
    ), time_elapsed

def blindspot_with_prune(model, unlearning_teacher, retain_train_dl, retain_valid_dl, forget_train_dl,
                         forget_valid_dl, valid_dl, device, weights_path, **kwargs):

    start = time.time()
    KL_temperature = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #lr=0.0001,
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
    model.train()

    ########
    with_l1 = True
    no_l1_epochs = 0  # "non l1 epochs"
    unlearn_lr = 0.01  # "initial learning rate"
    unlearn_epochs = 1  # "number of total epochs for unlearn to run"
    alpha = 5e-3  # "unlearn noise"

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
            output = model(x)
            optimizer.zero_grad()
            loss = UnlearnerLoss(output=output, labels=y, full_teacher_logits=full_teacher_logits,
                                 unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature,)

            ########
            if epoch < unlearn_epochs - no_l1_epochs:
                current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))  # decaying
                ## current_alpha = args.alpha * (epoch / (args.unlearn_epochs-args.no_l1_epochs))  # increasing
            elif unlearn_epochs - no_l1_epochs == 0:
                current_alpha = alpha
            else:
                current_alpha = 0
            if with_l1:
                loss += current_alpha * l1_regularization(model)

            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        loss = np.mean(losses)
        print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))

    end = time.time()
    time_elapsed = end - start
    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
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

    for x, label in forget_train_dl.dataset:
        unlearning_trainset.append((x, random.choice(unlearninglabels)))

    for x, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle_gtsrb(
        int(para2), model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=para1, forget_loader=forget_train_dl,
        mask=mask, dataname='GTSRB'
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
        forget_valid_dl,
        valid_dl,
        device,
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
            input, target = batch

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
        forget_valid_dl,
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
        weights_path=None,
    **kwargs,
):

    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, orig_target in tqdm(train_loader):
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
    alpha = 1e-8
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
        forget_valid_dl,
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
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )
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

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
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
    mask=None,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
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
        forget_valid_dl,
        valid_dl,
        device,
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

def train_negrad_1(model, model_init, retain_loader, forget_loader, loss_fn, optimizer, alpha, valid_loader=None,
                   mask=None, prune=False, epoch=0, dataname='GTSRB'):
    # MAE = nn.L1Loss()
    model.train()
    for idx, (batch_retain, batch_forget) in enumerate(zip(retain_loader, cycle(forget_loader))):
    # for idx, (batch_forget) in enumerate(forget_loader):
        batch_retain = [tensor.to(next(model.parameters()).device) for tensor in batch_retain]
        batch_forget = [tensor.to(next(model.parameters()).device) for tensor in batch_forget]
        input_r, target_r = batch_retain
        input_f, target_f = batch_forget
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
        if mask:
            # print("negative grad's mask is available!")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        # if dataname == 'GTSRB' and idx > 800:
        #     break

        if dataname == 'GTSRB' and idx%10==0:
            with torch.no_grad():
                model.eval()
                print(f"Epoch[{epoch}]:", "Retain Dataset Acc", evaluate_gtsrb(model, valid_loader,
                                                                               next(model.parameters()).device), ",Forget Dataset Acc",
                      evaluate_gtsrb(model, forget_loader, next(model.parameters()).device))
                forget_acc = evaluate_gtsrb(model, forget_loader, next(model.parameters()).device)["Acc"]

                if forget_acc < 1:
                    return

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
        input_r, target_r = batch_retain
        input_f, _ = batch_forget

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
    for fs, labels in (dataloader):
        batch_fs = fs
        break
    for ii in range(num_classes):
        F[ii] = [torch.zeros_like(Classifier(batch_fs[:1].to(device))[0, :]).to(device)]
    batch_size = len(batch_fs)
    # print("batch_size", batch_size)
    # print("F[ii]", F[ii][0].shape)
    for fs, labels in (dataloader):
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
    para1=0.01,
    para2=2,
    **kwargs,
):
    alpha = 0.95
    #the defending para
    # epochs = 3#5
    # lr = 0.01#0.0001#0.01
    #the baseline para
    epochs = int(para2)
    lr = float(para1)#0.0001
    quiet = True
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)#always the Adam?

    #useless
    model_init = copy.deepcopy(model)
    # num_classes = 19
    # feature_ini = ref_f(forget_train_dl, model_init, num_classes, device)
    start = time.time()
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_1(model, model_init, retain_train_dl, forget_train_dl, loss_fn, optimizer, alpha, valid_loader=valid_dl,
                       mask=mask, epoch=epoch, dataname='GTSRB')

    end = time.time()
    time_elapsed = end - start

    torch.save(model.state_dict(), weights_path)
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    ), time_elapsed

def negative_grad_with_prune(
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
    **kwargs,
):
    alpha = 0.5#0.95
    epochs = 5
    lr = 0.01
    quiet = True
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        train_negrad_1(model, model_init, retain_train_dl, forget_train_dl, loss_fn, optimizer, alpha, mask, valid_loader=valid_dl, prune=True, epoch=epoch)

    torch.save(model.state_dict(), weights_path)
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

    for i, (image, target) in enumerate(val_loader):
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
        para1=0.01,
        para2=1,
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

        for i, (image, target) in enumerate(retain_train_dl):
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
        forget_valid_dl,
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
        data, labels = batch
        model.zero_grad()
        data = data.to(device)
        label = labels.to(device)
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
        data, labels = batch
        model.zero_grad()
        data = data.to(device)
        label = labels.to(device)
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
        para1=50,
        mask=None,
        **kwargs,
):
    alpha = para1
    print("alpha:", alpha)

    device = next(model.parameters()).device
    print("device:", device)

    batch_size = 32
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
        data, labels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = labels.to(device)
        output = model(data)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, batch in enumerate(tqdm(retain_grad_loader)):
        data, labels = batch
        model.zero_grad()
        real_num = data.shape[0]
        data = data.to(device)
        label = labels.to(device)
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
    if mask:
        print("mask is available")
        curr = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                length = param.view(-1).shape[0]
                perturb[curr: curr + length] = (perturb[curr: curr + length].view(param.shape)*mask[name]).view(-1)
                curr += length

    apply_perturb(model, alpha * perturb)

    # defence
    # _ = fit_one_cycle_gtsrb(
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
        forget_valid_dl,
        valid_dl,
        device,
    ), time_elapsed