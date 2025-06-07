<div align="center">
<img src="figs/logo.png" width="180px">


![](https://img.shields.io/badge/version-v0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/last_commit-May-blue)
![](https://img.shields.io/badge/PRs-Welcome-red)

---
 
<p align="center">
  <a href="#requirements">Installation</a> •
  <a href="#use-easyedit">QuickStart</a> •
    <a href="https://zjunlp.gitbook.io/easyedit">Doc</a> •
    <a href="https://arxiv.org/abs/2401.01286">Paper</a> •
    <a href="https://huggingface.co/spaces/zjunlp/EasyEdit">Demo</a> •
    <a href="https://huggingface.co/datasets/zjunlp/KnowEdit">Benchmark</a> •
  <a href="#contributors">Contributors</a> •
  <a href="https://github.com/zjunlp/EasyEdit/blob/main/tutorial.pdf">Slides</a> •
    <a href="https://youtu.be/Gm6T0QaaskU", target="_blank">Video</a> •
   <a href="https://twitter.com/_akhaliq/status/1742371655765164133", target="_blank">Featured By AK</a>
</p>
</div>

<h1 style="font-size: 32px; color: #333; text-align: left;">
  <span style="font-size: 30px; color:rgb(180, 148, 61);">📢</span> Update
</h1>
<p align="left">
  🔥 <strong><a href="README_2.md" style="text-decoration: none; ">EasyEdit 2.0</a></strong> officially published! <br>
 Unlike EasyEdit1.0, which achieves knowledge editing by updating internal parameters or introducing additional parameters, EasyEdit2 enables real-time steering of LLMs during inference. It provides a unified framework for controllability without retraining, seamlessly integrating various steering methods for ease of use and evaluation.
  &nbsp; 👉 <strong><a href="README_2.md" style="text-decoration: none; ">[README]</a></strong> (for more details). 
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [🔔News](#news)
- [Editing Demo](#editing-demo)
- [Knowledge Editing](#knowledge-editing)
  - [Task Definition](#task-definition)
    - [Knowledge insert](#knowledge-insert)
    - [Knowledge update](#knowledge-update)
    - [Knowledge erase](#knowledge-erase)
  - [Comparisons of the different technologies](#comparisons-of-different-technologies)
  - [Evaluation](#evaluation)
- [🌟Overview](#overview)
    - [Current Implementation](#current-implementation)
    - [Quick Start on Some Works](#quick-start-on-some-works)
    - [Tutorial notebook](#tutorial-notebook)
- [Requirements](#requirements)
    - [🔧Pip Installation](#pip-installation)
    - [Editing GPU memory usage](#editing-gpu-memory-usage)
- [📌Use EasyEdit](#use-easyedit)
  - [BaseEditor](#baseeditor)
    - [Introduction by a Simple Example](#introduction-by-a-simple-example)
  - [Evaluation](#evaluation-1)
  - [Trainer](#trainer)
- [Use EasyEdit with KnowEdit](#use-easyedit-with-knowedit)
  - [Dataset](#dataset)
  - [Usage](#usage)
- [Editing Performance](#editing-performance)
- [Citation](#citation)
- [🎉Contributors](#contributors)
    - [Other Related Projects](#other-related-projects)

## 🔔News
- 2025-06-07, 👑 [UltraEdit](https://github.com/XiaojieGu/UltraEdit) has arrived — powered by a lifelong normalization strategy that continuously updates feature statistics across turns, it can edit 20K samples on a 7B model in just 5 minutes and scales stably to millions !
- 2025-06-05, 🌟🌟the EasyEdit has added a new model editing algorithm [CORE](https://arxiv.org/abs/2505.23026), designed to strengthen context robustness by minimizing context-sensitive variance in hidden states of the model for edited knowledge. 
- 2025-05-28, 🌟🌟the EasyEdit has added a new model editing algorithm [NAMET](https://arxiv.org/abs/2505.11876), which introduces noise during memory extraction via a one-line modification to MEMIT. Thanks to [@ybdai7](https://github.com/ybdai7) for contribution!
- 2025-05-15, 🚀🚀We released a new blog [Reflection on Knowledge Editing: Charting the Next Steps](https://fish-sorrel-a54.notion.site/Reflection-on-Knowledge-Editing-Charting-the-Next-Steps-1e6ca8e41f3a8098bd14c85ac1db8da6) discussing the next step for knowledge editing research.
- 2025-04-03, 🚀🚀We have upgraded **EasyEdit** and introduced **EasyEdit2**! EasyEdit2 provides an easy-to-use steering framework for editing large language models with **precision and flexibility**. This upgrade enhances the framework’s capabilities, making it more robust and adaptable for various steering methods. More details can be found in our [paper](https://arxiv.org/abs/2504.15133), in the [README](README_2.md) and on our [website](https://zjunlp.github.io/project/EasyEdit2/).


<details>
<summary><b>Previous News</b></summary>

- 2025-03-04, 🌟🌟In addition to the original token-level teacher-forcing paradigm for evaluation, EasyEdit has integrated a new evaluation method, following the paper titled "[The Mirage of Model Editing: Revisiting Evaluation in the Wild](https://arxiv.org/abs/2502.11177)". You can use this [script](https://github.com/zjunlp/EasyEdit/blob/main/examples/run_LLM_evaluation.py) to quickly launch this evaluation approach, which better aligns with real-world requirements. Special thanks to [@WanliYoung](https://github.com/WanliYoung) for contribution!
- 2025-01-03, We have updated the evaluation method in paper "[CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs](https://arxiv.org/abs/2409.05806)", adopting an LLM-as-a-judge approach that is more aligned with real-world scenarios and practical applications. Both the experimental results and the running scripts have been updated accordingly in [readme](https://github.com/zjunlp/EasyEdit/blob/main/examples/CKnowEdit.md).
- 2024-11-19, we update the Table 4 results in the paper "[A Comprehensive Study of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2401.01286)" after optimizing certain methods (related to AdaLoRA) and fixing computational bugs (related to ROME and MEMIT) in the EasyEdit (More details in https://github.com/zjunlp/EasyEdit/issues/427). These improvements have led to better results than before. We will continue updating this paper and welcome everyone to discuss and exchange ideas.
- 2024-11-11, 🎉🎉the paper on model editing for LLMs4Code, "[Model Editing for LLMs4Code: How Far are We?](https://arxiv.org/abs/2411.06638)", has been accepted by ICSE 2025! This work proposes a benchmark for LLMs4Code editing, [CLMEEval](https://github.com/xpq-tech/code-llmedit), which is built upon EasyEdit!
- 2024-11-09, we fixed a bug regarding the [KnowEdit](https://huggingface.co/datasets/zjunlp/KnowEdit) results in the https://github.com/zjunlp/EasyEdit/issues/390. Thanks for the help of [@StarLooo](https://github.com/StarLooo) to help us with it.

- 2024-10-24, EasyEdit has added two new knowledge editing methods, [AlphaEdit](https://github.com/zjunlp/EasyEdit/blob/main/easyeditor/models/alphaedit/README.md). In addition, we have fixed several bugs.
- 2024-10-23, the EasyEdit integrates constrained decoding methods from steering editing to mitigate hallucination in LLM and MLLM, with detailed information available in [DoLa](https://github.com/zjunlp/EasyEdit/tree/main/easyeditor/models/dola) and [DeCo](https://github.com/zjunlp/EasyEdit/tree/main/easyeditor/models/deco).
- 2024-09-26, 🎉🎉 our paper "[WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/abs/2405.14768)" has been accepted by **NeurIPS 2024**.
- 2024-09-20, 🎉🎉 our papers: "[Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/abs/2407.15017)" and "[Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/abs/2403.06259)" have been accepted by **EMNLP 2024 Findings**.
  
- 2024-07-29, the EasyEdit has added a new model editing algorithm [EMMET](https://arxiv.org/abs/2403.14236), which generalizes ROME to the batch setting. This essentially allows making batched edits using the ROME loss function.

- 2024-07-23, we release a new paper: "[Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/abs/2407.15017)", which reviews how knowledge is acquired, utilized, and evolves in large language models. This survey may provide the fundamental mechanisms for precisely and efficiently manipulating (editing) knowledge in LLMs.

- 2024-06-04, 🎉🎉 [EasyEdit Paper](https://arxiv.org/abs/2308.07269) has been accepted by the **ACL 2024** System Demonstration Track.

- 2024-06-03, we released a paper titled **["WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models"](https://arxiv.org/abs/2405.14768)**, along with introducing **a new editing task: [Continuous Knowledge Editing](#continuous-knowledge-editing)** and correspondding **lifelong editing method** called [WISE](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md).

- 2024-04-24, EasyEdit announced support for the **ROME method for [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)**. Users are advised to update their transformers package to version 4.40.0.

- 2024-03-29, EasyEdit introduced **rollback support for GRACE**. For a detailed introduction, refer to the [EasyEdit documentation](#use-easyedit). Future updates will gradually include rollback support for other methods.

- 2024-03-22, a new paper titled **"[Detoxifying Large Language Models via Knowledge Editing](https://arxiv.org/abs/2403.14472)"** was released, along with a new dataset named [SafeEdit](https://huggingface.co/datasets/zjunlp/SafeEdit) and a new **detoxification method** called [DINM](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md).

- 2024-03-12, another paper titled **"[Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/abs/2403.06259)"** was released, introducing a new dataset named [ConceptEdit](https://huggingface.co/datasets/zjunlp/ConceptEdit).

- 2024-03-01, EasyEdit added support for a new method called **FT-M**. This method involves training a specific MLP layer **using cross-entropy loss on the target answer and masking the original text**. It outperforms the **FT-L** implementation in [ROME](https://github.com/kmeng01/rome). The author of issue https://github.com/zjunlp/EasyEdit/issues/173  is thanked for their advice.

- 2024-02-27, EasyEdit added support for a new method called [InstructEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md), with technical details provided in the paper **"[InstructEdit: Instruction-based Knowledge Editing for Large Language Models](https://arxiv.org/abs/2402.16123)"**.
<!-- - **2024-02-20 The AAAI2024 tutorial "*Knowledge Editing for Large Language Models*" has been canceled since speakers cannot present in person, we make this ppt[[Github](https://github.com/zjunlp/KnowledgeEditingPapers/blob/main/AAAI2024%40Tutorial_Knowledge%20Editing%20for%20LLMs.pdf)] [[Google Drive](https://drive.google.com/file/d/1fkTbVeRJSWmU7fBDeNf1OhHEkLSofQde/view?usp=sharing)] [[Baidu Pan](https://pan.baidu.com/s/1oJYgaMnxWIBE4kIcJuMSKg?pwd=p9j5)] available to the community**. -->
  
- 2024-02-09, the EasyEdit has added the support for the Dynamic LoRA model editing method [MELO'AAAI24](https://arxiv.org/abs/2312.11795).
- 2024-02-06, we release a new paper: "[EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models](https://arxiv.org/abs/2402.03049)" with an HF demo [EasyInstruct](https://huggingface.co/spaces/zjunlp/EasyInstruct).
- 2024-02-06, we release a preliminary tool [EasyDetect](https://github.com/OpenKG-ORG/EasyDetect) for LLM hallucination detection，with a [demo](http://easydetect.openkg.cn/).
- 2024-01-24, the EasyEdit has added the support for editing [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) (manually update transformers==4.34.0), we have also fixed some bugs in evaluating MEND (slightly influence the performance).
- 2024-01-16, the EasyEdit has added the support for the precise model editing method [PMET'AAAI24](https://arxiv.org/abs/2308.08742).
- 2024-01-03, we release a new paper:"[A Comprehensive Study of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2401.01286)" with a new benchmark [KnowEdit](https://huggingface.co/datasets/zjunlp/KnowEdit)! KnowEdit is constructed by re-organizing and cleaning existing datasests including WikiBio, ZsRE, WikiData Counterfact, WikiData Recent, convsent, Sanitation with new train/val/test spliting. Special thanks to the builders and maintainers of the those datasets.We are looking forward to any comments or discussions on this topic :)
- 2023-12-06, the EasyEdit has added the support for the lifelong model editing method [GRACE'NeurIPS24](https://arxiv.org/abs/2211.11031).
- 2023-11-18, our tutorial "Knowledge Editing for Large Language Models" has been accepted by COLING 2024.
- 2023-10-25, our tutorial "Knowledge Editing for Large Language Models" has been accepted by AAAI 2024.
- 2023-10-24, the EasyEdit has added the support for efficient editing of [Baichuan2](https://github.com/baichuan-inc/Baichuan2), [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B), [InternLM](https://github.com/InternLM/InternLM), [QWen](https://github.com/QwenLM/Qwen) and fixed several bugs for a better user experience.
- 2023-10-14, we release the [MultimodalEditor](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md) based on the paper "[Can We Edit Multimodal Large Language Models?](https://arxiv.org/abs/2310.08475)".
- 2023-10-13, we release the paper "[Can We Edit Multimodal Large Language Models?](https://arxiv.org/abs/2310.08475)" accepted by EMNLP 2023.
- 2023-10-08, our paper "[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)" has been accepted by EMNLP 2023.
- 2023-10-07, the EasyEdit has added the support for editing models with multiple GPUs, using huggingface [`Accelerate`](https://github.com/zjunlp/EasyEdit/blob/main/hparams/ROME/llama-7b.yaml#L24).
- 2023-9-21, the EasyEdit has added the support for Parameter-Efficient Fine-Tuning through AdaLoRA to inject knowledge into the LLM.
- 2023-8-31, the EasyEdit has added the support for official fine-tuning API for gpt-3.5-turbo to customize ChatGPT for your editing cases.
- 2023-8-15, we release the paper "[EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269)."
- 2023-7-12, we release version 0.0.1, supporting several knowledge editing techniques for LLMs. EasyEdit helps to better align LLMs with changing needs and values of users.
- 2023-5-22, we release the paper "[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)" and provide a paper list at [PaperList](https://github.com/zjunlp/KnowledgeEditingPapers).
- 2023-3-25, the EasyEdit project has been launched and is under development.

</details>

<!-- **EasyEdit** is now publicly open-sourced, with a [demo video](https://www.youtube.com/watch?v=NaQRvSYuQMo) and long-term maintenance. -->

---

> A Comprehensive Study of Knowledge Editing for Large Language Models [[paper](https://arxiv.org/abs/2401.01286)][[benchmark](https://huggingface.co/datasets/zjunlp/KnowEdit)][[code](https://github.com/zjunlp/EasyEdit)] 

> IJCAI 2024 Tutorial [Google Drive](https://drive.google.com/file/d/1sg3Dy3gQo2bwOoLwWMFh57NfHROf3Ahg/view?usp=sharing)

> COLING 2024 Tutorial [Google Drive](https://drive.google.com/file/d/1vFzRYjnzkuZaNdjdIxQwWbEybCY7YqY9/view?usp=sharing)  

> AAAI 2024 Tutorial [Google Drive](https://drive.google.com/file/d/1fkTbVeRJSWmU7fBDeNf1OhHEkLSofQde/view?usp=sharing) 

> AACL 2023 Tutorial [[Google Drive](https://drive.google.com/file/d/1EW-cusC_llCM0wEshkIdYuYrvfBPCDRz/view?usp=sharing)] [[Baidu Pan](https://pan.baidu.com/s/1NupastGJUzcUIAjI64J1tw?pwd=i5an)]


## Editing Demo

There is a demonstration of editing. The GIF file is created by [Terminalizer](https://github.com/faressoft/terminalizer). <br>

We provide a handy [Jupyter Notebook](https://github.com/zjunlp/EasyEdit/blob/main/tutorial-notebooks/EasyEdit_Example_US_President.ipynb)! It allows you to edit a LLM's knowledge of the US president, switching from Biden to Trump and even back to Biden. This includes methods like [WISE](https://arxiv.org/abs/2405.14768), [AlphaEdit](https://arxiv.org/abs/2410.02355), AdaLoRA, and Prompt-based editing.


<img src="figs/demo_usage_new.gif" width="550" height="470" align=center>

## Knowledge Editing

<div align=center><img src="./figs/ke.png" width="100%" height="80%" /></div>



### Task Definition

Deployed models may still make unpredictable errors. For example, LLMs notoriously _hallucinate_, _perpetuate bias_, and _factually decay_, so we should be able to adjust specific behaviors of pre-trained models.

**Knowledge editing** aims to adjust base model's $(f_\theta)$ behavior on the particular edit descriptor $[x_e, y_e]$​​​ efficiently.

### Multi Setting

#### Single Knowledge Editing

Evaluating the performance of the model after a single edit. The model reloads the original weights (e.g. LoRA discards the adapter weights) after a single edit. You should set **`sequential_edit=False`**

$$\theta' \leftarrow \text{arg} \min\limits_{\theta} (\Vert f_\theta(x_e) - y_e \Vert)$$

#### Continuous Knowledge Editing

This requires **sequentially editing**, and evaluation is performed after all knowledge updates have been applied:

$$\theta' \leftarrow \text{arg} \min\limits_{\theta} \sum_{e=1}^{\Vert X_e \Vert} (\Vert f_\theta(x_e) - y_e \Vert)$$

It makes parameter adjustments for $(x_e, y_e)$, where $x_e \in X_e$ and $f_\theta'(x_e) = y_e$. Here, $X_e$​ represents the whole **edit set**. To enable continuous editing, you can set **`sequential_edit=True`**: [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md) (for more details).

### Multi Scenario

<details><summary> <b> Factual Knowledge Editing </b> </summary>

##### Knowledge insert

- Inject knowledge that LLMs have not seen before. such as:
  - *How many times has Messi won the World Cup? 0* $\rightarrow$ **1**:


##### Knowledge update

- Update outdated knowledge. such as:
  - *The president of USA: Donald Trump* $\rightarrow$ **Joe Biden**:


##### Knowledge erase

- Erase sensitive information. such as:
  - *The phone number of someone is XXXX* $\rightarrow$ **__**

Without influencing the model behavior on unrelated samples, the ultimate goal is to create an edited model $(f_\theta')$​​.

</details>

<details><summary> <b> Safety Editing </b> </summary>
**Detoxifying LLM** strives to build a safe and trustworthy large language model (LLM). Knowledge editing focuses on specific areas for permanent adjustment without compromising overall performance. Then, detoxifying LLM via knowledge editing leverages a small amount of data, usually an instance, to correct the toxic behaviors of the LLM. The edited LLM can defend against various malicious inputs. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)
</details>

<details><summary> <b> MultiModal Model Editing </b> </summary>

Editing Task for *Image Captioning* and *Visual Question Answering*. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md)
</details>

<details><summary> <b> Personality Editing </b> </summary>

The proposed task takes the preliminary attempt to edit LLMs' personalities by editing their opinions on specific topics, given that an individual's opinions can reflect aspects of their personality traits. We draw upon the established [BIG FIVE theory](https://en.wikipedia.org/wiki/Big_Five_personality_traits) as a basis for constructing our dataset and assessing the LLMs' personality expressions. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/PersonalityEdit.md)

**Evaluation**

Logits-based

- **ES**: evaluating the editing success rate based on the logits of pre-generated text.
- **DD**: evaluating whether the model changes opinions on other topics based on the logits of pre-generated text.

Generation-based

- **Acc**: the accuracy of the generated text after editing the model on target personality.
- **TPEI**: measuring whether generated opinion text from the edited model leans more towards the target personality.
- **PAE**: utilizing GPT-4 to evaluate the personality traits in generated text.

While for assessing **Acc** and **TPEI**, you can download the trained classifier from [here](https://huggingface.co/shai-msy/per-classifier).

</details>


### Comparisons of different technologies

<div align=center><img src="./figs/comparison.png" width="60%" height="48%" /></div>

### Evaluation

The knowledge editing process generally impacts the predictions for a broad set of inputs **that are closely** associated with the edit example, called the **editing scope**.

A successful edit should adjust the model’s behavior within the editing scope while remaining unrelated inputs:

$$
f_{\theta_{e}}(x) = \begin{cases}
y_e & \text{if } x \in I(x_e,y_e) \\
f_{\theta}(x) & \text{if } x \in O(x_e, y_e) \end{cases}
$$

- `Reliability`: the success rate of editing with a given editing descriptor
- `Generalization`: the success rate of editing within the editing scope
- `Locality`: whether the model's output changes after editing for unrelated inputs
- `Portability`: the success rate of editing for reasoning/application(one hop, synonym, logical generalization)
- `Efficiency`: time and memory consumption


## 🌟Overview

EasyEdit is a Python package for edit Large Language Models (LLM) like `GPT-J`, `Llama`, `GPT-NEO`, `GPT2`, `T5`(support models from **1B** to **65B**), the objective of which is to alter the behavior of LLMs efficiently within a specific domain without negatively impacting performance across other inputs. It is designed to be easy to use and easy to extend.

<h3 align="center">
<img src="figs/FrameWork.png">
</h3>

- EasyEdit contains a unified framework for **Editor**, **Method** and **Evaluate**, respectively representing the editing scenario, editing technique, and evaluation method.

- Each Knowledge Editing scenario comprises of three components:

  - `Editor`: such as BaseEditor(**Factual Knowledge** and **Generation** Editor) for LM, MultiModalEditor(**MultiModal Knowledge**).
  - `Method`: the specific knowledge editing technique used(such as **ROME**, **MEND**, ..).
  - `Evaluate`: **Metrics** for evaluating knowledge editing performance.
    - `Reliability`, `Generalization`, `Locality`, `Portability`

- The current supported knowledge editing techniques are as follows:
  
  
  - Memory-based: [SERAC](https://github.com/eric-mitchell/serac), [IKE](https://github.com/Zce1112zslx/IKE), [GRACE](https://github.com/thartvigsen/grace), [MELO](https://github.com/ECNU-ICALK/MELO), [WISE](https://arxiv.org/abs/2405.14768)
  - Meta-learning: [MEND](https://github.com/eric-mitchell/mend), [InstructEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md), [MALMEN](https://github.com/ChenmienTan/malmen)
  - Locate-then-edit: [KN](https://github.com/Hunter-DDM/knowledge-neurons), [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit), [PMET](https://github.com/xpq-tech/PMET), [DINM](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md), [R-ROME](https://github.com/scalable-model-editing/rebuilding-rome), [EMMET](https://github.com/scalable-model-editing/unified-model-editing)
  - [FT-L](https://github.com/kmeng01/rome)
  > Note 1: Due to the limited compatibility of this toolkit, some knowledge editing methods including  [T-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher), [KE](https://github.com/nicola-decao/KnowledgeEditor), [CaliNet](https://github.com/dqxiu/CaliNet)
  > are not supported. 
  >
  > Note 2: Similarly, the [MALMEN](https://github.com/ChenmienTan/malmen) method is only partially supported due to the same reasons and will continue to be improved.
#### Current Implementation

You can choose different editing methods according to your specific needs. GPT series includes GPT-2, GPT-J and GPT-NEO.
| **Method** | T5 | GPT series | LlaMA | Baichuan | ChatGLM | InternLM | Qwen | Mistral
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| FT | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AdaLoRA |  |  | ✅ |  | ✅ | | | |
| SERAC | ✅ | ✅ | ✅ |  | |  | | |
| IKE | ✅ | ✅ | ✅ | ✅ |✅  | ✅ | ✅ | ✅ |
| MEND | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| KN  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROME | | ✅ | ✅ | ✅ |✅ | ✅ | ✅ | ✅ |
| r-ROME | | ✅ | ✅ | ✅ |✅ | ✅ | ✅ | ✅ |
| MEMIT | | ✅ | ✅ | ✅ | ✅| ✅ | ✅ | ✅ | |
| EMMET | | ✅ | ✅ |    |   |    |    |    | |
| GRACE| | ✅ |  ✅|  |  |  | | |
| MELO | |✅ |  |  |  |  | | |
| PMET | |✅ |  ✅|  |  |  | | |
| [InstructEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md) | | ✅   |  ✅|  |  |  | | |
| [DINM](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)| |✅ |✅  |  |  |  | | ✅|
| [WISE](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md) | |✅ |✅  | ✅ |  |  |✅ | |
| Defer | | ✅ |✅  |  |  |  | | ✅ |
| [AlphaEdit](https://github.com/zjunlp/EasyEdit/blob/main/easyeditor/models/alphaedit/README.md) | |✅ |✅  |  |  |  | | |
| [UltraEdit](https://github.com/guangxuc42/update-UltraEdit/blob/main/examples/UltraEdit.md) | |✅ |✅  |  |  |  |✅ | ✅ |

> ❗️❗️ If you intend to use Mistral, please update the `transformers` library to version 4.34.0 manually. You can use the following code: `pip install transformers==4.34.0`.

We also support some knowledge editing methods for LMMs (Large Multimodal Models). Please refer to [MultimodalEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/MultimodalEdit.md) for more details.
| **Method** | MiniGPT-4 | BLIP-2 | LLaVA-OneVision | Qwen2-VL |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| MEND | ✅ | ✅ |  |  |
| SERAC | ✅ | ✅ |  |  |
| IKE |  |  | ✅ | ✅ |
| LoRA |  |  | ✅ | ✅ |
| GRACE |  |  | ✅ | ✅ |
| WISE |  |  | ✅ | ✅ |


<!-- | **Method** | Model Name | Description |
| :--------: | :--------: | :--------: | 
| [FT-Api](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates) | [gpt-3.5-turbo(ChatGPT)](https://github.com/zjunlp/EasyEdit/blob/main/hparams/FT-Api/gpt-3.5-turbo.yaml) | official fine-tuing Api for gpt-3.5-turbo | -->



#### Quick Start on Some Works

| **Work** | Description |    Path   |
| :--------: | :---------: | :-------: |
|InstructEdit|InstructEdit: Instruction-based Knowledge Editing for Large Language Models| [Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md)|
|DINM|Detoxifying Large Language Models via Knowledge Editing|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)|
|WISE|WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md)|
|ConceptEdit|Editing Conceptual Knowledge for Large Language Models|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/ConceptEdit.md)|
|MMEdit|Can We Edit Multimodal Large Language Models?|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md)|
|PersonalityEdit|Editing Personality For Large Language Models|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/PersonalityEdit.md)|
|PROMPT|PROMPT-based knowledge editing methods|[Quick Start](https://github.com/zjunlp/EasyEdit/blob/main/examples/PROMPT.md)|

### Dataset

**Benchmark: KnowEdit** [[Hugging Face]](https://huggingface.co/datasets/zjunlp/KnowEdit)[[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit)[[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit)
> ❗️❗️ To be noted, **KnowEdit** is constructed by **re-organizing and extending** existing datasests including **WikiBio**, **ZsRE**, **WikiData<sub>Counterfact</sub>**,  **WikiData<sub>Recent</sub>**, **convsent**, **Sanitation** to make a comprehensive evaluation for knowledge editing. Special thanks to the builders and maintainers of the those datasets.

> Please note that Counterfact and WikiData<sub>Counterfact</sub> are not the same dataset.

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Task</th>
    <th class="tg-7btt">Knowledge Insertion</th>
    <th class="tg-7btt" colspan="4">Knowledge Modification</th>
    <th class="tg-7btt">Knowledge Erasure</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Datasets</td>
    <td class="tg-c3ow">Wiki<sub>recent</sub></td>
    <td class="tg-c3ow">ZsRE</td>
    <td class="tg-c3ow">WikiBio</td>
    <td class="tg-c3ow"> WikiData<sub>counterfact</sub></td>
    <td class="tg-c3ow">Convsent</td>
    <td class="tg-c3ow">Sanitation</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Type</td>
    <td class="tg-c3ow">Fact</td>
    <td class="tg-c3ow">Question Answering</td>
    <td class="tg-c3ow">Hallucination</td>
    <td class="tg-c3ow">Counterfact</td>
    <td class="tg-c3ow">Sentiment</td>
    <td class="tg-c3ow">Unwanted Info</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Train</td>
    <td class="tg-c3ow">570</td>
    <td class="tg-c3ow">10,000</td>
    <td class="tg-c3ow">592</td>
    <td class="tg-c3ow">1,455</td>
    <td class="tg-c3ow">14,390</td>
    <td class="tg-c3ow">80</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Test</td>
    <td class="tg-c3ow">1,266</td>
    <td class="tg-c3ow">1301</td>
    <td class="tg-c3ow">1,392</td>
    <td class="tg-c3ow">885</td>
    <td class="tg-c3ow">800</td>
    <td class="tg-c3ow">80</td>
  </tr>
</tbody>
</table>

We provide **detailed scripts** for user to easily use KnowEdit, please refer to [examples](https://github.com/zjunlp/EasyEdit/blob/main/examples/KnowEdit.md).

<details><summary> <b> dataset description </b> </summary>

- ZsRE: is a context-free question-answering task. Given a question based on the subject and relation, the model is expected to provide the correct object as the answer. 
- Wiki<sub>recent</sub>: This dataset specifically focuses on triplets that have been recently inserted into WikiData after July 2022. 
- WikiBio: The original dataset was created by prompting GPT-3 to generate 238 Wikipedia-style biographies using subjects from the WikiBio.
- WikiData<sub>counterfact</sub>: Since tail entities are often not captured by models, and therefore are not suitable for testing modification edits, RippleEdit collects triplets about popular entities, where the subject corresponds to one of the top-viewed pages in Wikipedia.
- Convsent: This is a sentiment editing task that assesses the model's ability to modify a dialog agent's sentiment on a specific topic without affecting its responses to other topics.
- Sanitation: This dataset specifically addresses privacy concerns associated with learned language models. 
</details>


<details><summary> <b> dataset structure </b> </summary>

```text
knowedit
├── WikiBio
│   ├── wikibio-test-all.json
│   └── wikibio-train-all.json
├── ZsRE
│   └── ZsRE-test-all.json
├── wiki_counterfact
│   ├── test_cf.json
│   └── train_cf.json
├── convsent
│   ├── blender_test.json
│   ├── blender_train.json
│   └── blender_val.json
├── convsent
│   ├── trivia_qa_test.json
│   └── trivia_qa_train.json
└── wiki_recent
    ├── recent_test.json
    └── recent_train.json
```

</details>

---
#### Datasets for Chinese Knowledge: CKnowEdit

| **dataset** | HuggingFace| WiseModel | ModelScope | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |:--------------------------------------------------------------------------------: |
| CKnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/CKnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/CKnowEdit) | [[ModelScope]](https://modelscope.cn/datasets/ZJUNLP/CKnowEdit) | dataset for editing Chinese Knowledge |

- Here, you can follow [CKnowEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/CKnowEdit.md) to find more details about **CKnowEdit** and run Chinese knowledge editing experiments.

<details><summary> <b> dataset description </b> </summary>

**CKnowEdit** is a high-quality Chinese-language dataset for knowledge editing which is highly characterized by the Chinese language, with all data sourced from Chinese knowledge bases. It is meticulously designed to more deeply discern the nuances and challenges inherent in the comprehension of the Chinese language by current LLMs, providing a robust resource for refining Chinese-specific knowledge within LLMs.

The field descriptions for the data in **CKnowEdit** are as follows:

```python
"prompt": query inputed to the model (str)
"target_old": the incorrect response previously generated by the model (str)
"target_new": the accurate answer of the prompt (str)
"portability_prompt": new prompts related to the target knowledge (list or None)
"portability_answer": accurate answers corresponding to the portability_prompt (list or None)
"locality_prompt": new prompts unrelated to the target knowledge (list or None)
"locality_answer": accurate answers corresponding to the locality_prompt (list or None)
"rephrase": alternative ways to phrase the original prompt (list)
```
</details>

<details><summary> <b> dataset structure </b> </summary>

```text
CknowEdit
├── Chinese Literary Knowledge
│   ├── Ancient Poetry
│   ├── Proverbs
│   └── Idioms
├── Chinese Linguistic Knowledge
│   ├── Phonetic Notation
│   └── Classical Chinese
├── Chinese Geographical Knowledge
└── Ruozhiba
```
</details>

---

#### Datasets for Factual Knowledge
| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| _ZsRE_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Question Answering dataset using question rephrasings |
| _Counterfact_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Counterfact dataset using Entity replacement |


We provide zsre and counterfact datasets to verify the effectiveness of knowledge editing. You can download them here. [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing), [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky).

- For **locality**, in addition to testing unrelated instances, we also provide tests on distracting ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553)), other attribution, and other downstream tasks (such as commonsense reasoning).
- For **portability**, it tests whether the model can apply edited instances for inference. We provide evaluations for one-hop reasoning, subject alias, and inverse relation (eg, a one-to-one relationship between spouses should be bidirectionally edited).

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── counterfact
│   ├── counterfact-edit.json
│   ├── counterfact-train.json
│   └── counterfact-val.json
├── locality
│   ├── Commonsense Task
│   │   ├── piqa_valid-labels.lst
│   │   └── piqa_valid.jsonl
│   ├── Distracting Neighbor
│   │   └── counterfact_distracting_neighbor.json
│   └── Other Attribution
│       └── counterfact_other_attribution.json
├── portability
│   ├── Inverse Relation
│   │   └── zsre_inverse_relation.json
│   ├── One Hop
│   │   ├── counterfact_portability_gpt4.json
│   │   └── zsre_mend_eval_portability_gpt4.json
│   └── Subject Replace
│       ├── counterfact_subject_replace.json
│       └── zsre_subject_replace.json
└── zsre
    ├── zsre_mend_eval.json
    ├── zsre_mend_train_10000.json
    └── zsre_mend_train.json
```

- counterfact: original counterfact dataset using Entity replacement
- zsre: original question answering dataset using question rephrasings
- locality (evaluation for locality, see details in this [paper](https://arxiv.org/abs/2305.13172))
    - Commonsense Task: evaluation for other downstream tasks such as commonsense task
    - Distracting Neighbor: test on distracting neighborhood ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553))
    - Other Attribution
- portability
    - Inverse Relation: evaluation for one-to-one relationship such as `spouse`
    - One Hop: evaluation for one-hop reasoning
    - Subject Replace: evaluation for synonym replacement
    </details>

---

#### Datasets for Conceptual Knowledge: ConceptEdit

| **dataset** | Google Drive| HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| ConceptEdit | [[Google Drive]](https://drive.google.com/drive/folders/1Hp1DfIuj6Ih6ZLVENS-UmgJT8mRBlFC2?usp=drive_link) |[[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/ConceptEdit) | dataset for editing conceptual knowledge |

- Here, you can follow [ConceptEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/ConceptEdit.md) to run concept editing experiments.
  
<details><summary> <b> dataset description </b> </summary>

```text
data
└──concept_data.json
    ├──final_gpt2_inter.json
    ├──final_gpt2_intra.json
    ├──final_gptj_inter.json
    ├──final_gptj_intra.json
    ├──final_llama2chat_inter.json
    ├──final_llama2chat_intra.json
    ├──final_mistral_inter.json
    └──final_mistral_intra.json
```

**Concept Specific Evaluation Metrics**

- `Instance Change`: capturing the intricacies of these instance-level changes
- `Concept Consistency`: the semantic similarity of generated concept definition
</details>

---

#### Datasets for Multimodal Knowledge: MMEdit

| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| E-IC | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Image Captioning_ |
| E-VQA | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Visual Question Answering_ |

- All **images** used in **E-IC** and **E-VQA** are available for download at [Google Drive](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view)
- For **locality**, it is the same as factual editing in order to measure whether unrelated facts retain their outputs.
- For **multimodal locality**, it assesses the impact of editing on the visual module, which is similar to regular **locality**.

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── caption
│   ├── caption_train_edit.json
│   └── caption_eval_edit.json
├── locality
│   ├── NQ dataset
│   │   ├── train.json
│   │   └── validation.json
├── multimodal_locality
│   ├── OK-VQA dataset
│   │   ├── okvqa_loc.json
└── vqa
    ├── vqa_train.json
    └── vqa_eval.json
```
- Multimodal locality (evaluation for multimodal locality, see dataset's details in this [paper](http://openaccess.thecvf.com/content\_CVPR\_2019/html/Marino\_OK-VQA\_A\_Visual\_Question\_Answering\_Benchmark\_Requiring\_External\_Knowledge\_CVPR\_2019\_paper.html)) 
</details>

---
#### Datasets for detoxifying LLMs: SafeEdit

| **dataset** | HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| SafeEdit |[[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) | dataset for detoxifying LLMs |

- Here, you can follow [SafeEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md) to run detoxification editing experiments.
  
<details><summary> <b> dataset description </b> </summary>

```text
data
└──SafeEdit_train.json
└──SafeEdit_val.json
└──SafeEdit_test.json
    
```

**Detoxifying Specific Evaluation Metrics**
- `Defense Duccess (DS)`: the detoxification success rate of edited LLM for adversarial input (attack prompt + harmful question), which is used to modify LLM.
- `Defense Generalization (DG)`: the detoxification success rate of edited LLM for out-of-domain malicious inputs.
- `General Performance`: the side effects for unrelated task performance.
</details>

#### Tutorial notebook

| **Method** |          Description           |                                                 GPT-2                                                 |                                           LlaMA                                            |
| :--------: | :----------------------------: | :---------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
|   _IKE_    | In-Context Learning (ICL) Edit |       [[Colab-gpt2]](https://colab.research.google.com/drive/1m6Xg05XCs_WZKH0D9KJQqg9z0ZiDhEkL)       | [[Colab-llama]](https://colab.research.google.com/drive/1m6Xg05XCs_WZKH0D9KJQqg9z0ZiDhEkL) |
|   _ROME_   |    Locate-Then-Edit Neurons    | [[Colab-gpt2]](https://colab.research.google.com/drive/1KkyWqyV3BjXCWfdrrgbR-QS3AAokVZbr?usp=sharing) | [[Colab-llama]](https://colab.research.google.com/drive/1W18GPlBCV9K6lDy7eX8V5W0knTLr5r0A) |
|  _MEMIT_   |    Locate-Then-Edit Neurons    |       [[Colab-gpt2]](https://colab.research.google.com/drive/1P1lVklP8bTyh8uxxSuHnHwB91i-1LW6Z)       | [[Colab-llama]](https://colab.research.google.com/drive/19fKCKtVBU2fqj6eTvDokGoTrxvXkEPPq) |


---
#### Datasets for Real-World Lifelong Knowledge Editing: WikiBigEdit

| **dataset** | HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| WikiBigEdit |[[HuggingFace Dataset]](https://huggingface.co/datasets/lukasthede/WikiBigEdit) | 500k knowledge edits from Wikidata |

- Here, you can follow [run_WikiBigEdit.py](https://github.com/zjunlp/EasyEdit/blob/main/examples/run_WikiBigEdit.py) to run **WikiBigEdit** edits using EasyEdit.
  
- If you want to learn more about **WikiBigEdit**, please refer to the following paper: 
[Understanding the Limits of Lifelong Knowledge Editing in LLMs](https://arxiv.org/abs/2503.05683)

## Requirements

#### 🔧Pip Installation

**Note: Please use Python 3.9+ for EasyEdit**
To get started, simply install conda and run:

```shell
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
...
pip install -r requirements.txt
```


#### Editing GPU memory usage
Our results are all based on the default configuration

|         | llama-2-7B | chatglm2 | gpt-j-6b | gpt-xl |
|:-------:|:----------:|:--------:|:--------:|:------:|
|   FT    |    60GB    |   58GB   |   55GB   |  7GB   |
|  SERAC  |    42GB    |   32GB   |   31GB   |  10GB  |
|   IKE   |    52GB    |   38GB   |   38GB   |  10GB  |
|  MEND   |    46GB    |   37GB   |   37GB   |  13GB  |
|   KN    |    42GB    |   39GB   |   40GB   |  12GB  |
|  ROME   |    31GB    |   29GB   |   27GB   |  10GB  |
|  MEMIT  |    33GB    |   31GB   |   31GB   |  11GB  |
| AdaLoRA |    29GB    |   24GB   |   25GB   |  8GB   |
|  GRACE  |    27GB    |          |   23GB   |  6GB   |
|  WISE  |    34GB    |          |   27GB   |  7GB   |
<!-- editing multimodal -->
## 📌Use EasyEdit

- Edit large language models(LLMs) around **_5 seconds_**

- Following example shows you how to perform editing with EasyEdit. More examples and tutorials can be found at [examples](https://github.com/zjunlp/EasyEdit/tree/main/examples)

### BaseEditor

> `BaseEditor`is the class for Language Modality Knowledge Editing. You can choose the appropriate editing method based on your specific needs.

- Due to different transformer versions and different GPU models, the editing results may fluctuate **slightly**.

#### Introduction by a Simple Example

With the modularity and flexibility of `EasyEdit`, you can easily use it to edit model.

**Step1: Define a PLM as the object to be edited.**
Choose the PLM to be edited. `EasyEdit` supports partial models(`T5`, `GPTJ`, `GPT-NEO`, `LlaMA` so far) retrievable on [HuggingFace](https://huggingface.co/). The corresponding configuration file directory is `hparams/YUOR_METHOD/YOUR_MODEL.YAML`, such as `hparams/MEND/gpt2-xl.yaml`, set the corresponding `model_name` to select the object for knowledge editing.

```yaml
model_name: gpt2-xl
model_class: GPT2LMHeadModel
tokenizer_class: GPT2Tokenizer
tokenizer_name: gpt2-xl
model_parallel: false # true for multi-GPU editing
```

**Step2: Choose the appropriate Knowledge Editing Method**

```python
## In this case, we use MEND method, so you should import `MENDHyperParams`
from easyeditor import MENDHyperParams
## Loading config from hparams/MEMIT/gpt2-xl.yaml
hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
```

**Step3: Provide the edit descriptor and edit target**

```python
## edit descriptor: prompt that you want to edit
prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
## You can set `ground_truth` to None !!!(or set to original output)
ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender']
## edit target: expected output
target_new = ['University of Michigan', 'Lamiinae', 'winger']
```

**Step4: Combine them into a `BaseEditor`**
`EasyEdit` provides a simple and unified way to init `Editor`, like huggingface: **from_hparams**.

```python
## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)
```

**Step5: Provide the data for evaluation**
Note that the data for portability and locality are both **optional**(set to None for basic editing success rate evaluation only). The data format for both is a **dict**, for each measurement dimension, you need to provide the corresponding prompt and its corresponding ground truth. Here is an example of the data:

```python
locality_inputs = {
    'neighborhood':{
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    },
    'distracting': {
        'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}
```

In the above example, we evaluate the performance of the editing methods about "neighborhood" and "distracting".

**Step6: Edit and Evaluation**
Done! We can conduct Edit and Evaluation for your model to be edited. The `edit` function will return a series of metrics related to the editing process as well as the modified model weights. [`sequential_edit=True` for continuous editing]

```python
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    sequential_edit=False # True: start continuous editing ✈️
)
## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model
```
The maximum input length for EasyEdit is 512. If this length is exceeded, you will encounter the error "CUDA error: device-side assert triggered." You can modify the maximum length in the following file:[LINK](https://github.com/zjunlp/EasyEdit/blob/7d947abfa2975dcdbbd81a355b8f69d47e1b421f/easyeditor/evaluate/evaluate_utils.py#L115)

**Step7: RollBack**
In sequential editing, if you are not satisfied with the outcome of one of your edits and you do not wish to lose your previous edits, you can use the rollback feature to undo your previous edit. Currently, we only support the GRACE method. All you need to do is a single line of code, using the edit_key to revert your edit.

```
editor.rolllback('edit_key')
```
In EasyEdit, we default to using target_new as the edit_key
### Evaluation

We specify the return metrics as `dict` format, including model prediction evaluations before and after editing. For each edit, it will include the following metrics:

- `rewrite_acc` $\rightarrow$ **Reliablilty**
- `rephrase_acc` $\rightarrow$ **Generalization**
- `locality` $\rightarrow$ **Locality**
- `portablility` $\rightarrow$ **Portablility**

```json
{
    "post": {
        "rewrite_acc": ,
        "rephrase_acc": ,
        "locality": {
            "YOUR_LOCALITY_KEY": ,
            //...
        },
        "portablility": {
            "YOUR_PORTABILITY_KEY": ,
            //...
        },
    },
    "pre": {
        "rewrite_acc": ,
        "rephrase_acc": ,
        "portablility": {
            "YOUR_PORTABILITY_KEY": ,
            //...
        },
    }
}
```

- For evaluation for Reliablilty, you only need to provide the corresponding editing `prompts` and editing `target_new`.
- For evaluation for Generalization, `rephrase_prompts` are required.
- For evaluation for Locality and Portablility, you need to define the name of the corresponding metric, as well as `prompts` and `ground_truth`.
  - > Note: the length needs to be equal to the edit prompts

### Trainer

- meta-learning based: `MEND`
- memory-based routing: `SERAC`

For above editing methods, pre-training of corresponding meta-networks or classifiers is required. Therefore, in EasyEdit, we provide a unified framework for pretraining the relevant network structures. Take the training MEND for example:

- **Step 1** and **Step 2** are the same as the example above, which involves selecting the appropriate editing model and editing method.

**Step3: Provide the edit training set**
The currently supported and available datasets are: `zsre` and `counterfact`([Google Drive](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing)). Please place them in the "data" directory and initialize the dataset_class (`ZsreDataset` for zsre and `CounterFactDataset` for counterfact) to load the corresponding training set.

```python
train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
```

**Step4: Combine them into a `Trainer`**

```python
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
```

**Step5: Run and Edit**
Done! We can conduct Run and Evaluation.

```python
trainer.run()
```

- Run: The `CHECKPOINT` will be saved to the path `results_dir`.
- Edit: Set the `archive` field in the **hparams file** to `CHECKPOINT`. EasyEdit will automatically load the corresponding pre-trained weights during the editing process([Go to edit](#use-easyedit)).

**Training Example**
```python
from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset

training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/llama-7b.yaml')
train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()
```

<!-- ## Overall Results
> Note that the following experimental results are from this [paper](https://arxiv.org/abs/2305.13172).The actual editing performance of this tool is still under testing and will be announced **as soon as possible**.
*  We tested the editing performance of different knowledge editing methods on various model, the test results are shown in the table below(`-` refers to the results that the methods empirically fail to edit LLMs). -->
<!--
- For `zsre` dataset:

<div style="text-align: center">
<table style="text-align: center">
    <tr>
        <th></th><th colspan="3" style="text-align: center;">T5-3B</th><th colspan="3" style="text-align: center;">GPT-J</th>
    </tr>
    <tr>
        <td><b>Method</b></td><td>Reliability</td><td>Generalization</td><td>Locality</td><td>Reliability</td><td>Generalization</td><td>Locality</td>
    </tr>
    <tr>
        <td>FT</td><td>20.71</td><td>19.68</td><td>89.01</td><td>54.70</td><td>49.20</td><td>37.24</td>
    </tr>
    <tr>
        <td>SERAC</td><td>99.80</td><td>99.66</td><td>98.13</td><td>90.16</td><td>89.96</td><td>99.90</td>
    </tr>
    <tr>
        <td>IKE</td><td>67.00</td><td>67.11</td><td>63.60</td><td>99.96</td><td>99.87</td><td>59.21</td>
    </tr>
    <tr>
        <td>KE</td><td>3.00</td><td>5.40</td><td>96.43</td><td>6.60</td><td>7.80</td><td>94.18</td>
    </tr>
    <tr>
        <td>MEND</td><td>78.80</td><td>89.80</td><td>98.45</td><td>45.60</td><td>48.00</td><td>88.21</td>
    </tr>
    <tr>
        <td>KN</td><td>22.51</td><td>22.70</td><td>16.43</td><td>11.34</td><td>9.40</td><td>90.03</td>
    </tr>
    <tr>
        <td>ROME</td><td>-</td><td>-</td><td>-</td><td>99.18</td><td>94.90</td><td>99.19</td>
    </tr>
    <tr>
        <td>MEMIT</td><td>-</td><td>-</td><td>-</td><td>99.23</td><td>87.16</td><td>99.62</td>
    </tr>
</table>
</div>

- For `counterfact` dataset:

<div style="text-align: center">
<table style="text-align: center">
    <tr>
        <th></th><th colspan="3" style="text-align: center;">T5-3B</th><th colspan="3" style="text-align: center;">GPT-J</th>
    </tr>
    <tr>
        <td><b>Method</b></td><td>Reliability</td><td>Generalization</td><td>Locality</td><td>Reliability</td><td>Generalization</td><td>Locality</td>
    </tr>
    <tr>
        <td>FT</td><td>33.57</td><td>23.54</td><td>72.72</td><td>99.90</td><td>97.53</td><td>1.02</td>
    </tr>
    <tr>
        <td>SERAC</td><td>99.89</td><td>98.71</td><td>99.93</td><td>99.78</td><td>99.41</td><td>98.89</td>
    </tr>
    <tr>
        <td>IKE</td><td>97.77</td><td>82.99</td><td>37.76</td><td>99.61</td><td>72.67</td><td>35.57</td>
    </tr>
    <tr>
        <td>KE</td><td>1.00</td><td>1.40</td><td>96.28</td><td>13.40</td><td>11.00</td><td>94.38</td>
    </tr>
    <tr>
        <td>MEND</td><td>81.40</td><td>93.40</td><td>91.58</td><td>73.80</td><td>74.20</td><td>93.75</td>
    </tr>
    <tr>
        <td>KN</td><td>47.86</td><td>46.78</td><td>57.10</td><td>1.66</td><td>1.38</td><td>58.28</td>
    </tr>
    <tr>
        <td>ROME</td><td>-</td><td>-</td><td>-</td><td>99.80</td><td>86.63</td><td>93.61</td>
    </tr>
    <tr>
        <td>MEMIT</td><td>-</td><td>-</td><td>-</td><td>99.90</td><td>73.13</td><td>97.17</td>
    </tr>
</table>
</div> -->

# Use EasyEdit with KnowEdit
## Dataset

KnowEdit is a benchmark dataset of knowledge editing for LLMs. You can easily obtain KnowEdit from HuggingFace, HuggingFace, and ModelScope.

| **dataset** | HuggingFace| HuggingFace | ModelScope |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| KnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/KnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit) | [[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit) |


## Usage 

We provide detailed scripts for user to easily use KnowEdit, please refer to [examples](https://github.com/zjunlp/EasyEdit/blob/main/examples/KnowEdit.md).

# Editing Performance

We present editing results of the four metrics on [LlaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) using EasyEdit. We adopt [ZsRE](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) as the test dataset.

> ❗️❗️Editing `llama-2-7B` requires 40G+ VRAM on GPU. (OOM [solution](https://github.com/zjunlp/EasyEdit/issues/9#issuecomment-1687284658))

|       | Reliability | Generalization |  Locality  | Portability |
| :---: | :---------: | :------------: | :--------: | :---------: |
| FT  |    56.94    |     52.02      |   96.32    |    51.03    |
| SERAC |    99.49    |     99.13      | **100.00** |    57.82    |
|  IKE  | **100.00**  |   **99.98**    |   69.19    |  **67.56**  |
| MEND  |    94.24    |     90.27      |   97.04    |    56.95    |
|  KN   |    28.95    |     28.43      |   65.43    |    37.18    |
| ROME  |    92.45    |     87.04      |   99.63    |    57.47    |
| MEMIT |    92.94    |     85.97      |   99.49    |    60.64    |



We also present editing results of KnowEdit on [LlaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) using EasyEdit. 

| DataSet                  | Metric        | SERAC  | ICE    | AdaLoRA | MEND   | ROME   | MEMIT  | FT-L   | FT-M   |
|--------------------------|---------------|--------|--------|---------|--------|--------|--------|--------|--------|
| **WikiData_recent**      |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 98.68  | 60.74  | 100.00  | 95.75  | 97.18  | 97.05  | 55.75  | 100.00 |
|                          | Portability | 63.52  | 36.93  | 64.69   | 55.88  | 55.25  | 56.37  | 40.86  | 65.44  |
|                          | Locality     | 100.00 | 33.34  | 56.42   | 94.76  | 54.77  | 52.15  | 43.70  | 64.33  |
|                          | Fluency     | 553.19 | 531.01 | 579.57  | 557.11 | 579.66 | 573.89 | 529.24 | 574.32 |
| **ZsRE**                 |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.67  | 66.01  | 100.00  | 96.74  | 96.77  | 95.37  | 53.93  | 99.98  |
|                          | Portability | 56.48  | 63.94  | 58.03   | 60.41  | 52.63  | 52.67  | 45.64  | 60.31  |
|                          | Locality   | 30.23  | 23.14  | 75.76   | 92.79  | 53.67  | 48.32  | 73.42  | 89.78  |
|                          | Fluency     | 410.89 | 541.14 | 563.56  | 524.33 | 573.75 | 563.31 | 493.01 | 552.26 |
| **WikiBio**              |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.69  | 95.53  | 100.00  | 93.66  | 96.08  | 94.40  | 66.33  | 100.00 |
|                          | Locality    | 69.79  | 47.90  | 81.28   | 69.51  | 62.74  | 61.51  | 79.86  | 93.38  |
|                          | Fluency    | 606.95 | 632.92 | 618.45  | 609.39 | 617.69 | 616.65 | 606.95 | 612.69 |
| **WikiData_counterfact** |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.99  | 69.83  | 100.00  | 80.03  | 98.57  | 98.05  | 45.15  | 100.00 |
|                          | Portability | 76.07  | 45.32  | 69.89   | 52.01  | 55.92  | 58.56  | 33.60  | 74.36  |
|                          | Locality    | 98.96  | 32.38  | 70.31   | 94.38  | 51.97  | 46.62  | 50.48  | 76.76  |
|                          | Fluency     | 549.91 | 547.22 | 580.29  | 555.72 | 584.04 | 575.96 | 528.26 | 575.62 |
| **ConvSent**             |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 62.75  | 52.78  | 44.89   | 50.76  | 45.79  | 44.75  | 49.50  | 46.10  |
|                          | Locality    | 0.26   | 49.73  | 0.18    | 3.42   | 0.00   | 0.00   | 0.00   | 0.00   |
|                          | Fluency     | 458.21 | 621.45 | 606.42  | 379.43 | 606.32 | 602.62 | 607.86 | 592.52 |
| **Sanitation**           |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 0.00   | 72.50  | 2.50    | 0.00   | 85.00  | 48.75  | 0.00   | 75.00  |
|                          | Locality    | 100.00 | 56.58  | 65.50   | 5.29   | 50.31  | 67.47  | 14.78  | 47.07  |
|                          | Fluency     | 416.29 | 794.15 | 330.44  | 407.18 | 465.12 | 466.10 | 439.10 | 416.29 |

> ❗️❗️ **Please note that if you wish to reproduce the results regarding Rome on Knowedit, ensure that `fp16: False`.**

> For the locality metric, we calculate the score based on the proportion of tokens that remain unchanged before and after editing. For example, if the output tokens before editing are [29, 234, 334] and after editing are [29, 234, 333], the locality score for this data would be 66.67. For the portability metric, we calculate it by taking the average of all sub-scores under the portability category.

<details><summary> <b> TO DO </b> </summary>
In next version, we plan to:

- Explore and integrate more robust editing methods, focusing on `locality` and `portability` metrics.
- Provide a comprehensive evaluation suite for editing methods, including fact modification, fact erasure and hallucination erasure.
- Provide a causal analysis component for analyzing knowledge storage mechanisms.
- knowledge editing for other tasks(except factual editing), like `personality editing`, etc.

Meanwhile, we will offer long-term maintenance to fix bugs, solve issues and meet new requests. So if you have any problems, please put issues to us.

</details>

# 🎊How to contribute to EasyEdit

## New Method

In **EasyEdit**, we uniformly use the `apply_algo` member function to call specific editing methods. You need to submit a pull request (PR) with the required files to EasyEdit according to the following steps ( The following uses **YourMethod** as the method name as an example ):

### (1) YourMethod_main.py

In this file, there must be a function named `apply_YourMethod_to_model` to serve as the interface function for the main program to call the editing method, and its parameter list should also follow the example below:

```
def apply_YourMethod_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:

# Here is the specific implementation of the method.
```

After completing the implementation of the editing process, you need to return the following at the end of the `apply_YourMethod_to_model` function:

```
return model, weights_copy
```

Here, `model` is the **edited model**, and `weights_copy` is the **original weights** corresponding to the modified parameter regions (used to restore the model's weights).

### (2) YourMethod_hparams.py

Define a class to describe the hyperparameters required for the implementation of your method：

```
class YourMethodHyperParams(HyperParams):
```

### ✨Additional Explanation

> You can add any necessary `.py` files for the implementation of your method.  

> Additionally, you will need an `__init__.py` file.  

> Finally, package all your `.py` files into a folder and submit a pull request (PR) to the `EasyEdit/easyeditor/models` directory.

## New Dataset

The `Field` settings of the dataset are closely related to the evaluation metrics. Currently, **EasyEdit** supports the evaluation metrics of **Reliability**, **Generalization**, **Portability**, and **Locality**. Therefore, you need to reorganize and rename the fields of your dataset to adapt to the current **EasyEdit**.

### YourDataset.py

This file is generally responsible for importing the dataset and making preliminary adjustments to the data `Fields`:

```
{
    "subject":record["subject"],
    "prompt": record["edit_prompt"],   
    "target_new": record["edit_target"],  
    "ground_truth": record["ground_truth"],
    "rephrase": record['rephrase'],
    "portability_type1": record["portability_type1"],
    "portability_type2": record["portability_type2"],
    ...
    "locality": record["loc"] if "loc" in record else None,
}
```

### ✨Additional Explanation

> Add a startup file under the `EasyEdit/examples` folder to further reorganize the fields. You can follow the process in the `examples/run_knowedit_llama2.py` file. 

> Add your dataset class in the `__init__.py` file under the `EasyEdit/dataset` folder.

> Finally submit a pull request (PR).

## Citation

Please cite our paper if you use EasyEdit in your work.

```bibtex

@article{zhang2024comprehensive,
  title={A Comprehensive Study of Knowledge Editing for Large Language Models},
  author={Zhang, Ningyu and Yao, Yunzhi and Tian, Bozhong and Wang, Peng and Deng, Shumin and Wang, Mengru and Xi, Zekun and Mao, Shengyu and Zhang, Jintian and Ni, Yuansheng and others},
  journal={arXiv preprint arXiv:2401.01286},
  year={2024}
}

@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

@article{cheng2023edit,
  title={Can We Edit Multimodal Large Language Models?}, 
  author={Cheng, Siyuan and Tian, Bozhong and Liu, Qingbin and Chen, Xi and Wang, Yongheng and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2310.08475},
  year={2023}
}

@article{mao2023editing,
  title={Editing personality for llms},
  author={Mao, Shengyu and Zhang, Ningyu and Wang, Xiaohan and Wang, Mengru and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02168},
  year={2023}
}

@article{wang2024wise,
  title={WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models},
  author={Wang, Peng and Li, Zexi and Zhang, Ningyu and Xu, Ziwen and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2405.14768},
  year={2024}
}
```

## 🎉Contributors

<a href="https://github.com/zjunlp/EasyEdit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/EasyEdit" />
</a>

We thank all the contributors to this project, more contributors are welcome!

#### Other Related Projects

- [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit)
- [UltraEdit](https://github.com/XiaojieGu/UltraEdit)
- [ROME](https://github.com/kmeng01/rome)
- [FastEdit](https://github.com/hiyouga/FastEdit)
- [GRACE](https://github.com/Thartvigsen/GRACE)
- [MELO](https://github.com/ECNU-ICALK/MELO)
- [PMET](https://github.com/xpq-tech/PMET)
- [VLKEB](https://github.com/VLKEB/VLKEB)
- [PitfallsKnowledgeEditing](https://github.com/zjunlp/PitfallsKnowledgeEditing)
- [BiasEdit](https://github.com/zjunlp/BiasEdit)
- [WikiLLM](https://github.com/laramohan/wikillm)
- [PEAK](https://github.com/mjy1111/PEAK)
- [Debugger](https://github.com/openai/transformer-debugger)
- [LTE](https://github.com/YJiangcm/LTE)
- [r-ROME](https://github.com/scalable-model-editing/rebuilding-rome)
- [dive-into-llms](https://github.com/Lordog/dive-into-llms)

🙌 We would like to express our heartfelt gratitude for the contribution of [FastEdit](https://github.com/hiyouga/FastEdit), [ROME](https://github.com/kmeng01/rome), [GRACE](https://github.com/Thartvigsen/GRACE), [MELO](https://github.com/ECNU-ICALK/MELO), [PMET](https://github.com/xpq-tech/PMET) to our project, as we have utilized portions of their source code in our project. Many thanks to all the colleagues in the community for submitting issues and providing technical support. Appreciation is also extended to all PR contributors, and issue feedback providers during the EasyEdit version iterations, especially ancelia06 for correcting the grammar of README.
