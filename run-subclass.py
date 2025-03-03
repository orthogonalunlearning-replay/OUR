import subprocess
import time

import config
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

delay_seconds = 20

DEVICE = '1'
seed = '42'

net = 'ResNet18'
dataset = 'Cifar20'
n_classes = '20'
# forget_class_list = ['30', '55', '72', '95']
# forget_class_list = ['4', '55', '72', '95']
all_class_list = ['4', '30', '55', '72', '95']
for i in all_class_list:
    forget_class_list = list(set(all_class_list) - set([i]))
    #pretrain_model
    masked = config.CHECKPOINT_PATH+f'/pretrain/ResNet18-Cifar20-20-ood{i}/best.pth'

    # machine unlearning baseline
    python_file = "forget_sub_class_main_cifar20.py"
    mu_method_list = [["Woodfisher", "105", "0"], ["amnesiac", "0.000025", "2"]]#[["negative_grad", "0.003", "1"]]  # [["amnesiac", "0.000025", "2"]]# ['retrain', '0', '0'] [[
    # "finetune", "0.1", "15"], ["negative_grad", "0.005", "2"], ["amnesiac", "0.0001", "2"], ["Woodfisher", "80", "0"]]
    # [["negative_grad", "0.005", "2"], ["amnesiac", "0.0001", "2"], ["Wfisher", "80", "0"], ["finetune", "0.1", "15"]]
    # [['retrain', '0', '0']]
    # ["retrain", "Wfisher", "finetune", "badteacher", "negative_grad"] ["finetune", '0.013', '10']

    for forget_class in forget_class_list:
        for mu_method, para1, para2 in mu_method_list:
            subprocess.call(["python", python_file, '-net', net,
                             '-dataset', dataset, '-superclasses', n_classes,
                             '-method', mu_method, '--forget_class', forget_class, '-weight_path', masked,
                             '-seed', seed, '--para1', para1, '--para2', para2, '--sub_ood_class', i])
            time.sleep(delay_seconds)

    # #relearning mia attack
    python_file = "mia_on_mu_main_cifar20_subclass.py"
    # # # python_file = 'data_process\\process_mia_data.py'
    num_ood_dataset = "1"
    unlearn_data_percent = "0.20"  # 0.03 for full-class
    relearning_lr = '0.01'
    # # para1_list=['0.03', '0.05', '0.08', '0.1']
    # # para2_list=['2']
    for forget_class in forget_class_list:
        for mu_method, para1, para2 in mu_method_list:
            subprocess.call(["python", python_file, '-net', net,
                             '-dataset', dataset,
                             '-method', mu_method, '--forget_class', forget_class,
                             '-weight_path', masked,
                             '-seed', seed, "--num_ood_dataset", num_ood_dataset,
                             '--unlearn_data_percent', unlearn_data_percent, '--para1', para1, '--para2', para2,
                             '--relearning_lr', relearning_lr, '--sub_ood_class', i])
            time.sleep(delay_seconds)


