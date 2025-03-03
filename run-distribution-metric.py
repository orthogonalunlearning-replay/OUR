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
n_classes = '15'
forget_class ='4'

pretrain_path = config.CHECKPOINT_PATH+'/pretrain/ResNet18-Cifar20-15/best.pth'

#feature visualization
python_file ="feature_visualization_metric_main.py"
# mu_method_list = [["finetune", "0.015", "10"], ["FT_prune", "0.01", "10"], ["retrain", "0.01", "10"],
#                   ["Woodfisher", "50", "0"], ["Woodfisher_our", "120", "0"], ["baseline", "0", "0"],
#                   ["amnesiac_our", "0.0025", "5"], ["amnesiac", "0.0001", "3"]]
mu_method_list= [["ssd_tuning_our", '0.005', '0.4']]
for mu_method, para1, para2 in mu_method_list:
    subprocess.call(["python", python_file, '-net', net,
                     '-dataset', dataset, '-classes', n_classes,
                     '-forget_class', forget_class, '-method', mu_method,
                     '-weight_path', pretrain_path, '-seed', seed,
                     "--para1", para1, "--para2", para2])
    time.sleep(delay_seconds)
