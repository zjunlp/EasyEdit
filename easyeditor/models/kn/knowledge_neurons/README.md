# knowledge-neurons

An open source repository replicating the 2021 paper *[Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696)* by Dai et al., and extending the technique to autoregressive models, as well as MLMs.

The Huggingface Transformers library is used as the backend, so any model you want to probe must be implemented there. 

Currently integrated models:
```python
BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
```

The technique from Dai et al. has been used to locate knowledge neurons in the huggingface bert-base-uncased model for all the head/relation/tail entities in the PARAREL dataset. Both the neurons, and more detailed results of the experiment are published at `bert_base_uncased_neurons/*.json` and can be replicated by running `pararel_evaluate.py`. More details in the `Evaluations on the PARAREL dataset` section. 

# Setup

Either clone the github, and run scripts from there:

```bash
git clone knowledge-neurons
cd knowledge-neurons
```

Or install as a pip package:

```
pip install knowledge-neurons
```

# Usage & Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EleutherAI/knowledge-neurons/blob/master/examples/knowledge_neurons.ipynb)

An example using bert-base-uncased:

```python
from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
import random

# first initialize some hyperparameters
MODEL_NAME = "bert-base-uncased"

# to find the knowledge neurons, we need the same 'facts' expressed in multiple different ways, and a ground truth
TEXTS = [
    "Sarah was visiting [MASK], the capital of france",
    "The capital of france is [MASK]",
    "[MASK] is the capital of france",
    "France's capital [MASK] is a hotspot for romantic vacations",
    "The eiffel tower is situated in [MASK]",
    "[MASK] is the most populous city in france",
    "[MASK], france's capital, is one of the most popular tourist destinations in the world",
]
TEXT = TEXTS[0]
GROUND_TRUTH = "paris"

# these are some hyperparameters for the integrated gradients step
BATCH_SIZE = 20
STEPS = 20 # number of steps in the integrated grad calculation
ADAPTIVE_THRESHOLD = 0.3 # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
P = 0.5 # the threshold for the sharing percentage

# setup model & tokenizer
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

# initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

# use the integrated gradients technique to find some refined neurons for your set of prompts
refined_neurons = kn.get_refined_neurons(
    TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
)

# suppress the activations at the refined neurons + test the effect on a relevant prompt
# 'results_dict' is a dictionary containing the probability of the ground truth being generated before + after modification, as well as other info
# 'unpatch_fn' is a function you can use to undo the activation suppression in the model. 
# By default, the suppression is removed at the end of any function that applies a patch, but you can set 'undo_modification=False', 
# run your own experiments with the activations / weights still modified, then run 'unpatch_fn' to undo the modifications
results_dict, unpatch_fn = kn.suppress_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons
)

# suppress the activations at the refined neurons + test the effect on an unrelated prompt
results_dict, unpatch_fn = kn.suppress_knowledge(
    "[MASK] is the official language of the solomon islands",
    "english",
    refined_neurons,
)

# enhance the activations at the refined neurons + test the effect on a relevant prompt
results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

# erase the weights of the output ff layer at the refined neurons (replacing them with zeros) + test the effect
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
)

# erase the weights of the output ff layer at the refined neurons (replacing them with an unk token) + test the effect
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="unk"
)

# edit the weights of the output ff layer at the refined neurons (replacing them with the word embedding of 'target') + test the effect
# we can make the model think the capital of france is London!
results_dict, unpatch_fn = kn.edit_knowledge(
    TEXT, target="london", neurons=refined_neurons
)
```

for bert models, the position where the `"[MASK]"` token is located is used to evaluate the knowledge neurons, (and the ground truth should be what the mask token is expected to be), but due to the nature of GPT models, the last position in the prompt is used by default, and the ground truth is expected to immediately follow.

In GPT models, due to the subword tokenization, the integrated gradients are taken n times, where n is the length of the expected ground truth in tokens, and the mean of the integrated gradients at each step is taken.

for bert models, the ground truth is currently expected to be a single token. Multi-token ground truths are on the todo list.

# Evaluations on the PARAREL dataset
To ensure that the repo works correctly, figures 3 and 4 from the knowledge neurons paper are reproduced below. In general the results appear similar, except suppressing unrelated facts appears to have a little more of an affect in this repo than in the paper's original results.* 

Below are Dai et al's, and our result, respectively, for suppressing the activations of the refined knowledge neurons in pararel:
![knowledge neuron suppression / dai et al.](images/suppress_original.png)
![knowledge neuron suppression / ours](images/suppress.png)

And Dai et al's, and our result, respectively, for enhancing the activations of the knowledge neurons:
![knowledge neuron enhancement / dai et al.](images/enhance_original.png)
![knowledge neuron enhancement / ours](images/enhance.png)

To find the knowledge neurons in bert-base-uncased for the PARAREL dataset, and replicate figures 3. and 4. from the paper, you can run
```bash
# find knowledge neurons + test suppression / enhancement (this will take a day or so on a decent gpu) 
# you can skip this step since the results are provided in `bert_base_uncased_neurons`
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py
# plot results 
python plot_pararel_results.py
```

*It's unclear where the difference comes from, but my suspicion is they made sure to only select facts with different relations, whereas in the plots below, only a different pararel UUID was selected. In retrospect, this could actually express the same fact, so I'll rerun these experiments soon.

# TODO:
- [ ] Better documentation
- [ ] Publish PARAREL results for bert-base-multilingual-uncased
- [ ] Publish PARAREL results for bert-large-uncased
- [ ] Publish PARAREL results for bert-large-multilingual-uncased
- [ ] Multiple masked tokens for bert models
- [ ] Find good dataset for GPT-like models to evaluate knowledge neurons (PARAREL isn't applicable since the tail entities aren't always at the end of the sentence)
- [ ] Add negative examples for getting refined neurons (i.e expressing a different fact in the same way)
- [ ] Look into different attribution methods (cf. https://arxiv.org/pdf/2010.02695.pdf)


# Citations
```bibtex
@article{Dai2021KnowledgeNI,
  title={Knowledge Neurons in Pretrained Transformers},
  author={Damai Dai and Li Dong and Y. Hao and Zhifang Sui and Furu Wei},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.08696}
}
```
