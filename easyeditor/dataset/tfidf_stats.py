import json
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from . import AttributeSnippets
from ..util.globals import *

REMOTE_IDF_URL = f"{REMOTE_ROOT_URL}/data/dsets/idf.npy"
REMOTE_VOCAB_URL = f"{REMOTE_ROOT_URL}/data/dsets/tfidf_vocab.json"


def get_tfidf_vectorizer(data_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    data_dir = Path(data_dir)

    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"
    if not (idf_loc.exists() and vocab_loc.exists()):
        collect_stats(data_dir)

    idf = np.load(idf_loc)
    with open(vocab_loc, "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    return vec


def collect_stats(data_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.
    Retrieved later when computing TF-IDF vectors.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"

    try:
        print(f"Downloading IDF cache from {REMOTE_IDF_URL}")
        torch.hub.download_url_to_file(REMOTE_IDF_URL, idf_loc)
        print(f"Downloading TF-IDF vocab cache from {REMOTE_VOCAB_URL}")
        torch.hub.download_url_to_file(REMOTE_VOCAB_URL, vocab_loc)
        return
    except Exception as e:
        print(f"Error downloading file:", e)
        print("Recomputing TF-IDF stats...")

    snips_list = AttributeSnippets(data_dir).snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)

    idfs = vec.idf_
    vocab = vec.vocabulary_

    np.save(data_dir / "idf.npy", idfs)
    with open(data_dir / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)
