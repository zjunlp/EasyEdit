# SPHERE
> **Code for ICLR 2026: Energy-Regularized Sequential Model Editing on Hyperspheres**

[![arXiv](https://img.shields.io/badge/arXiv-2510.01172-b31b1b.svg)](https://arxiv.org/abs/2510.01172)
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2510.01172.svg)](https://doi.org/10.48550/arXiv.2510.01172)


![alt text](../figs/sphere.png)
*Figure: (a) A weight matrix is viewed as a set of neurons (red dots) on a hypersphere. (b) Current SOTA methods introduce perturbations (blue triangles) that
interfere with the principal hyperspherical directions of pre-edit weights. (c) SPHERE projects new knowledge onto a sparse space complementary to the principal hyperspherical directions.*


## Quick Start
### Example: Editing Llama3-8B-Instruct on ZsRE with SPHERE
#### 1. Prepare the base model (AlphaEdit)
This example uses AlphaEdit as the base method. Please follow [AlphaEdit](EasyEdit/examples/AlphaEdit.md) to run the base method first so the null-space projection matrix is computed successfully.

#### 2. Run the script [run_SPHERE_editing.py](EasyEdit/examples/run_SPHERE_editing.py)

    bash examples/run_SPHERE_editing.sh

This command runs the SPHERE editing script with Llama3-8B-Instruct. Arguments:

- `--editing_method=SPHERE`: Specifies the name of the editing algorithm being used, which is SPHERE in this case.
- `--hparams_dir=../hparams/SPHERE/llama3-8b.yaml`: Points to the yaml file containing hyperparameters specific to the Llama-3-8B model.
- `--data_dir=../data/ZsRE`: Specifies the dataset name, in this case, "ZsRE".
- `--ds_size=100`: Sets the total number of editing samples to 100.
- `--data_type=ZsRE`: Defines the dataset type.
- `--sequential_edit`: Enables continual (sequential) editing.
- `--cumulative_ratio=0.5`: Suppresses the top 50% of principal directions of edited weights.
- `--suppression_strength=0.5`: Controls how strongly perturbation components along principal directions are removed.


## Applying SPHERE to Other Methods
You can integrate SPHERE into any other model editing methods (eg. MEMIT, LoRA, ROME...) using the sparse projection below:

```python
def sparse_projection(A, B, eta=0.5, alpha=0.5):
    """
    Sparse projection used by SPHERE.

    :param A: Weight matrix of the current model
    :param B: Update matrix to be applied
    :param eta: Cumulative ratio
    :param alpha: Suppression strength

    :return: (1) Projected update matrix B_proj;
             (2) Soft projection matrix P_soft;
             (3) Eigenvectors U used in the projection
    """
    print(f"Projection Start with Cumulative ratio: {eta} and Suppression strength: {alpha}")
    A_hat = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    C = (A_hat.T @ A_hat) / A_hat.size(0)
    eigvals, eigvecs = eigh(C)
    cumsum = torch.cumsum(eigvals.flip(0), 0)
    total = eigvals.sum()
    r = (cumsum / total <= eta).sum().item() + 1
    U = eigvecs[:, -r:]
    P_soft = torch.eye(A.size(1), device=A.device) - alpha * (U @ U.T)
    B_proj = B @ P_soft.T
    return B_proj, P_soft, U
```



## Citation
If you use this code, please cite our paper:
```bibtex
@misc{liu2025energyregularizedsequentialmodelediting,
      title={Energy-Regularized Sequential Model Editing on Hyperspheres}, 
      author={Qingyuan Liu and Jia-Chen Gu and Yunzhi Yao and Hong Wang and Nanyun Peng},
      year={2025},
      eprint={2510.01172},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.01172}, 
}
```



## Acknowledgment
Our code is based on [``MEMIT``](https://github.com/kmeng01/memit.git), [``EMMET``](https://github.com/scalable-model-editing/unified-model-editing.git) and [``AlphaEdit``](https://github.com/jianghoucheng/AlphaEdit.git).