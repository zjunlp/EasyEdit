# AlphaEdit
- Code for [``AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models``]

- AlphaEdit minimizes disruption to the preserved knowledge by projecting parameter perturbations onto the null space of its key matrices. It then removes the output error related to it from the current objective, allowing the model to focus solely on knowledge update without trade-off.  By leveraging the mathematical properties of matrix projection and null space, AlphaEdit ensures that the distribution of hidden representations within LLMs remains invariant after edits. This invariance allows post-edited LLMs to effectively handle both knowledge update and preservation simultaneously.
- AlphaEdit focuses on optimizing sequential editing from an objective standpoint. 




## Quick Start
### An example for editing Llama3 (8B) on ZsRE dataset using AlphaEdit
#### 1. Make sure that the projection matrix P has been downloaded from the [baidu_netdisk]( https://pan.baidu.com/s/1Unk3X6jl3LZw_OF5eLoEeA?pwd=mcaf ) into this address (EasyEdit/null_space_project.pt) beforehand to avoid double computation. 

Due to limitations in computational power and time, we currently only provide the projection matrix P for LLAMA3-8B, specifically for layers [4, 5, 6, 7, 8]. For other models or different layers, the corresponding projection matrix P needs to be calculated independently. We will strive to update our resources in this regard.

#### 2. Find the [script](EasyEdit/examples/run_AlphaEdit_editing.sh)
 
    bash examples/run_AlphaEdit_editing.sh

This command runs an editing script for the AlphaEdit algorithm using the Llama3-8b. Below are the explanations for each argument:

- `--editing_method=AlphaEdit`: Specifies the name of the editing algorithm being used, which is AlphaEdit in this case.
- `--hparams_dir=../hparams/AlphaEdit/llama3-8b.yaml`: Points to the yaml file containing hyperparameters specific to the Llama-3-8B model.
- `--data_dir=../data/ZsRE`: Specifies the dataset name, in this case, "ZsRE".
- `--ds_size=100`: Sets the total number of editing samples to 100.
- `--data_type=ZsRE`: Defines the dataset type. 
- `--sequential_edit`: indicates that the editing process is continual editing.



## 📖 Citation

If finding this work useful for your research, you can cite it as follows:


```bibtex
@Article{Fang_arXiv_2024_p2410.02355,
    author =   {Junfeng Fang and Houcheng Jiang and Kun Wang and Yunshan Ma and Xiang
             Wang and Xiangnan He and Tat-seng Chua},
    title =    {{AlphaEdit: Null-Space Constrained Knowledge Editing for Language
             Models}},
    journal =  {arXiv},
    year =     2024,
    pages =    {2410.02355},
    doi =      {10.48550/arXiv.2410.02355},
    abstract = {Large language models (LLMs) often exhibit hallucinations due to
             incorrect or outdated knowledge. Hence, model editing methods have
             emerged to enable targeted knowledge updates. To achieve this, a
             prevailing paradigm is the locating-then-editing approach, which first
             locates influential parameters and then edits them by introducing a
             perturbation. While effective, current studies have demonstrated that
             this perturbation inevitably disrupt the originally preserved
             knowledge within LLMs, especially in sequential editing scenarios. To
             address this, we introduce AlphaEdit, a novel solution that projects
             perturbation onto the null space of the preserved knowledge before
             applying it to the parameters. We theoretically prove that this
             projection ensures the output of post-edited LLMs remains unchanged
             when queried about the preserved knowledge, thereby mitigating the
             issue of disruption. Extensive experiments on various LLMs, including
             LLaMA3, GPT2-XL, and GPT-J, show that AlphaEdit boosts the performance
             of most locating-then-editing methods by an average of 36.4{\%} with a
             single line of additional code for projection solely.},
}
```
