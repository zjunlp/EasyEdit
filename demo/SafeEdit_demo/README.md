<div align="center">

**Detoxifying Large Language Models via Knowledge Editing**

![](https://img.shields.io/badge/version-v0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/PRs-Welcome-red)

---

<p align="center">
  <a href="#🔧 Pip Installation">Pip Installation</a> •
    <a href="#⏩ Run">Run</a> •
    <a href="#🎍 Demo">Demo</a> • 
    <a href="#📖-citation">Citation</a> •
    <a href="https://arxiv.org/abs/2403.14472">Paper</a> •
    <a href="https://zjunlp.github.io/project/SafeEdit">Website</a> 
</p>
</div>

## 🔧 Pip Installation


To get started, simply install conda and run:

```shell
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
conda activate EasyEdit
cd EasyEdit
pip install -r requirements.txt
cd demo/SafeEdit_demo
pip install -r requirements.txt
```

> ❗️❗️ If you intend to use Mistral, please update the `transformers` library to version 4.34.0 manually. You can use the following code: `pip install transformers==4.34.0`.

---

## ⏩ Run

We provide one ways for users to quickly get started with SafeEdit. You can either use the Gradio app based on your specific needs.

### Gradio App

We provide a Gradio app for users to quickly get started with SafeEdit. You can run the following command to launch the Gradio app locally on the port `7860` (if available).

```shell
python app.py
```

Then you can access the URL `127.0.0.1:7860` to quickly use SafeEdit in the demo.

And If you want to replace the vanilla model used in the demo, you can set the `hparams_path` in the `utils.py` file to change the model configuration.

---

## 🎍 Demo

Here is the demo introduction of detoxifying Mistral-7B-v0.1 on one A800 GPU by DINM. 

You can download the [demo video](https://github.com/zjunlp/EasyEdit/blob/main/figs/SafeEdit_demo.mp4) and use [SafeEdit_demo](https://github.com/zjunlp/EasyEdit/tree/main/demo/SafeEdit_demo) to get started quickly.

Taking the Mistral-7B-v0.1 model as an example: 

- Click the button **Edit**: DINM use an instace to locate and edit toxic regions of Mistral-7B-v0.1. Then, we can obtain the toxic layer of Mistral-7B-v0.1, and edited Mistral-7B-v0.1.

- Click the button **Generate** of Defense Success: Edited Mistral-7B-v0.1 generates response for adversarial input, which is used for Defense Success metric.

- Click the button **Generate** of Defense Generalization: Edited Mistral-7B-v0.1 generates response for out-of-domain malicous input, which is used for Defense Generalization metric.


<div align=center>

<img src="../../figs/SafeEdit_demo_gif.gif" width="70%" height="70%" />

</div>



---

## 📖 Citation

Please cite our paper if you use **SafeEdit**, **DINM-Safety-Classifier** and **DINM** in your work.

```bibtex
@misc{wang2024SafeEdit,
      title={Detoxifying Large Language Models via Knowledge Editing}, 
      author={Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen},
      year={2024},
      eprint={2403.14472},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
