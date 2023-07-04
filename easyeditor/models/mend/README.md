# MEND: Model Editing Networks using Gradient Decomposition

If you run into any issues with the code, you can open an issue and/or email me at `eric.mitchell@cs.stanford.edu`

## Setup

### Environment

This codebase uses Python 3.7.9. Other versions may work as well.

Create a virtualenv ([pyenv](https://github.com/pyenv/pyenv) can help with this)
and install the dependencies:

    $ python -m venv env
    $ source env/bin/activate
    (env) $ pip install -r requirements.txt

### Data

You can download the data needed for this project from
[this Google Drive link](https://drive.google.com/drive/folders/1jAqBE45jEKR-5pMkwxlVQ0V8eKxqWbxA?usp=sharing).
Unzip each sub-directory into `mend/data` and you should be good to go.

## Running the code

Run MEND training/evaluation for distilGPT-2 on the wikitext editing problem with:

    (env) $ python -m run +alg=mend +experiment=gen +model=distilgpt2 data.wiki_webtext=False

Other valid algs include `efk` ([KnowledgeEditor](https://arxiv.org/abs/2104.08164))
and `enn` ([Editable Neural Networks](https://arxiv.org/abs/2004.00345)). Valid experiments
include `fc` (FEVER fact checking) and `qa` (zsRE question-answering). Splits and rephrases
for both come from [De Cao et. al](https://arxiv.org/abs/2104.08164). Check `config/model`
for options for editable models (note that all models don't work for all experiments; GPT-style
models only work with `gen`, seq2seq models only work with `qa`, and BERT only works with `fc`).

Also note that in the paper, we sample locality data from different datasets depending on the model.
By default, training will use [Natural Questions](https://ai.google.com/research/NaturalQuestions)
data (not zsRE data) for computing drawdown in the `qa` experiment and
[OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/). For models such as the `distilgpt2`
model we use (which was fine-tuned on wikitext) or the BART-base model, this behavior should be
disabled with `data.wiki_webtext=False` or `data.zsre_nq=False`, respectively.

## Citing the paper

If this code or paper was useful, please consider using the following citation:

    @article{mitchell2021fast,
        title={Fast Model Editing at Scale},
        author={Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Finn, Chelsea and Manning, Christopher D.},
        year={2021},
        journal={CoRR},
        url={https://arxiv.org/pdf/2110.11309.pdf}
    }      
