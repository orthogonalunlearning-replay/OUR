# Over-Unlearning and Replay

Welcome to the official repository for the paper **"How to Forget Data without A Trace: Over-Unlearning and Replay Are All You Need"**. This research explores a critical vulnerability in class-wise approximate unlearning algorithms and introduces the **Reminiscence Attack**, targeting the membership privacy of unlearned data.

## 1. Code Overview

The provided codebase enables:
1. **Estimation of distribution attributes** across various approximate unlearning models.
2. **Implementation of the Reminiscence Attack** on models processed by approximate unlearning algorithms.
3. **Execution of the Over-Unlearning and Replay framework** for these algorithms.

### (1). Pretraining

**To train a normal model:**  
Begin with pretraining a model using `pretrain_model_{dataset}.py`. 

**To train a sparse model:**  
Modify the training loss function to incorporate an L1 regularization for sparsity. Exclude out-of-distribution (OOD) data by setting the `ood_classes` in `config.py`.

### (2). Running Unlearning Algorithms

Post-pretraining, execute unlearning algorithms by running `run.py` with the following specifications:
- `dataset`: Options include `CIFAR20`, `CIFAR100`, `Imagenet64`.
- `model`: Specify the model to use.
- `total_number_of_classes`: Total classes in the dataset.
- `forget_class_index`: Index of the class to be forgotten.
- `pretrained_model_path`: Path to the pretrained model.
- `mu_method_list`: the configuration for recommended unlearning algorithms and parameters.

This project includes Python scripts designed for conducting machine unlearning under different scenarios. 

The `python file` are named following the pattern `forget_{full_class/subclass}_main_{dataset}.py`. Select the appropriate script according to the specifics of your experiment.

**Use `forget_full_class_main_{dataset}.py` for a complete class unlearning session.**

- **Standard Algorithms**: Use the standard method names (e.g., `Wfisher`).
- **Over-Unlearning & Replay**: Append `_our` to the method name (e.g., `Wfisher_our`).

Results are saved in the `./log_files` directory.

### (3). Distribution Metrics Analysis

Analyze the distribution of unlearned models by running `run-distribution-metric.py`. This script outputs various metrics such as Intra-class Variance, Silhouette Score, Overlap Score, KDE-estimated Overlap, and t-SNE visualizations.

### (4). Reminiscence Attack

#### White-Box Scenario

To perform the Reminiscence Attack in a white-box scenario:

1. Run the script `run.py` with the same parameters used in the unlearning phase.
2. Change the `python_file` parameter to `mia_on_mu_main_{dataset}_{fullclass/subclass}.py`.

#### Black-Box Scenario

To perform the Reminiscence Attack in a black-box scenario:

1. Initially, execute a Data-Free Model Extraction (DFME) attack using the script `dfme_attack.py` to obtain a copied model.
2. For the Reminiscence Attack on the copied model:
   - Set `python_file` parameter to `mia_on_mu_main_{dataset}_{fullclass/subclass}.py` in `run.py`.
   - Update the `checkpoint_path` in the script to point to the copied model's checkpoint.
   - Alternatively, use the `python_file` named with a `blackbox` suffix that automatically configures the necessary parameters.
  
### (5). Optimization Strategies

- [**L1 regularization**](https://github.com/OPTML-Group/Unlearn-Sparse): Add `l1_regularization` to the loss term. This parameter is defined in `forget_full_class_strategies.py`.
- [**L2 regularization**](https://github.com/cleverhans-lab/unrolling-sgd): Add `l2_penalty` to the loss term. This parameter is also defined in `forget_full_class_strategies.py`.
- [**SalUn**](https://github.com/OPTML-Group/Unlearn-Saliency): This requires three steps:
  1. First, generate masks by running `python_file = "saliency_mu\\generate_mask.py"` in `run.py`, specifying `masked_save_dir`.
  2. Then, pass `masked_save_dir` into the `--mask_path` parameter of the `forget_full_class_main_{dataset}.py` file.
  3. If `mask_path` is not None, SalUn strategies will be enabled, as already integrated in the code.


### (6). Acknowledgments

This project builds upon the outstanding work done in the following repositories:

- [Selective Synaptic Dampening (SSD)](https://github.com/if-loops/selective-synaptic-dampening)

We gratefully acknowledge the authors of the project for their significant contributions to the field and for making their code available to the community.
