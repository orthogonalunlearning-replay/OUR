import subprocess
import time

import config
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

delay_seconds = 20

DEVICE = '1'
seed = '42'

net = 'ResNet18'
dataset = 'Cifar20'     #chose from ['Cifar20', 'Cifar100']
n_classes = '15'        #chose from ['15', '95']
forget_class = '4'

pretrain_path = config.CHECKPOINT_PATH+'/pretrain/ResNet18-Cifar20-15/best.pth'
'''Optimization Strategy: SalUn'''
# python_file = "saliency_mu\\generate_mask.py"
# masked_save_dir = config.CHECKPOINT_PATH+'/masked/ResNet18-Cifar20-15'
#
# subprocess.call(["python", python_file, '--net', net, '--dataset', dataset, '--classes', n_classes,
#                      '--forget_class', forget_class, '--save_dir', masked_save_dir, '--mask', pretrain_path,
#                      '--seed', seed])
# time.sleep(delay_seconds)

'''machine unlearning baseline'''
python_file = "forget_full_class_main_cifar20.py"
mu_method_list = [["baseline", "0", "0"], ["retrain", "0.01", "40"],
                  ["negative_grad", "0.01", "2"], ["negative_grad_our", "0.01", "2"],
                  ["amnesiac", "0.0001", "3"], ["amnesiac_our", "0.0025", "0"],
                  ["Woodfisher", '50', '0'],
                  ["badteacher", "0.0001", "3"], ["badteacher_our", "0.0001", "3"],
                  ["ssd_tuning", '1', '1'], ["ssd_tuning_our", '0.005', '0.4']]

#for Wfisher_our, sparse pretrained model is necessary. Then, run
#mu_method_list = [["Wfisher_our", "120", "0"]]

for mu_method, para1, para2 in mu_method_list:
    subprocess.call(["python", python_file, '-net', net,
                    '-dataset', dataset, '-classes', n_classes,
                    '-method', mu_method, '--forget_class', forget_class,
                     '-weight_path', pretrain_path,
                    '-seed', seed, "--para1", para1, "--para2", para2])
    time.sleep(delay_seconds)

'''the model extraction attack for black-box scenarios'''
# python_file ="dfme_attack.py"
# for mu_method, para1, para2 in mu_method_list:
#     subprocess.call(["python", python_file, '-net', net,
#                      '-dataset', dataset, '-classes', n_classes,
#                      '-method', mu_method,
#                      '-forget_class', forget_class, '-weight_path', pretrain_path,
#                      '-seed', seed, "--para1", para1, "--para2", para2
#                      ])
# python_file = "mia_on_mu_main_cifar15_fullclass-blackbox.py"
# relearning mia attack masked = config.CHECKPOINT_PATH+'/model_extraction/Cifar20/cifar_dfme.pth.tar'

'''reminiscence attack'''
python_file = "mia_on_mu_main_cifar20_fullclass.py"
num_ood_dataset = "5"
unlearn_data_percent = "0.03"
lr_list = ['0.05', '0.10']
for mu_method, para1, para2 in mu_method_list:
    for lr in lr_list:
        subprocess.call(["python", python_file, '-net', net,
                         '-dataset', dataset, '-classes', n_classes,
                         '-method', mu_method, '--forget_class', forget_class,
                         '-weight_path', pretrain_path,
                         '-seed', seed, "--num_ood_dataset", num_ood_dataset,
                         '--unlearn_data_percent', unlearn_data_percent, '--para1', para1, '--para2', para2,
                         '--relearning_lr', lr])
        time.sleep(delay_seconds)
