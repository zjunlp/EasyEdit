## Dataset and Rules

#### Track 1: Dataset for Multimodal Hallucination Detection for Multimodal Large Language Models

You can download the datasets via [this link](https://huggingface.co/datasets/openkg/MHaluBench).

The expected structure of files is:

```
data
├── train.json                     # training dataset
├── val.json                       # validation dataset
├── test.json                      # test dataset which we will release in the future
```

> ❗️❗️**Data Utility Rules:** 
Due to the use of open source data, we do not provide image data. You need to download [MSCOCO-train2014](http://images.cocodataset.org/zips/train2014.zip), [MSCOCO-val2014](http://images.cocodataset.org/zips/val2014.zip), [TextVQA-train](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and [TextVQA-test](https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip) by yourself. For model training, **only the data provided by [this link](https://huggingface.co/datasets/openkg/MHaluBench) is allowed to be used as supervised data, which includes train.json, val.json.**    test.json will be used to evaluate the hallucination detected model or pipeline.

For more information related to this dataset, please refer to our paper: [Unified Hallucination Detection for Multimodal Large Language Models](https://arxiv.org/abs/2402.03190). 

#### Track 2: Dataset for Detoxifying Large Language Models

You can download the datasets via [this link](https://huggingface.co/datasets/zjunlp/SafeEdit).

The expected structure of files is:

```
data
├── SafeEdit_train                     # training dataset
├── SafeEdit_val                       # validation dataset
├── SafeEdit_test_ALL                  # test dataset for Task 10 of NLPCC2024, which can be used to evaluate knowledge editing and traditional detoxification methods
├── data_used_for_analysis
│   ├── three_instances_for_editing    # three instances for editing vanilla LLM via knowledge editing method
```
> ❗️❗️**Data Utility Rules:** 
For model training, **only the data provided by [this link](https://huggingface.co/datasets/zjunlp/SafeEdit) is allowed to be used as supervised data, which includes SafeEdit_train, SafeEdit_val, three_instances_for_editing.** 
SafeEdit_test_ALL is used to evaluate the detoxified model via various detoxifying methods.
SafeEdit_test_ALL and any variations of it cannot be used during the training phase.
Note that SafeEdit_test in [this link](https://huggingface.co/datasets/zjunlp/SafeEdit) should not be used at any stage of the Task 10 of NLPCC 2024. 




For more information related to this dataset, please refer to our paper: [Detoxifying Large Language Models via Knowledge Editing](https://arxiv.org/abs/2403.14472). 
If there are any differences between the paper and this page, the content of this page should prevail.
