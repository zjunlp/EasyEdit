import datetime
import getpass
import logging
import math
import os
import struct
import typing
from collections import defaultdict
import torch.nn as nn

import numpy as np
import torch

LOG = logging.getLogger(__name__)


def _inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


def shift_targets(config):
    return "t5" not in config.model_name.lower()


def scr():
    if os.path.exists("/scr-ssd"):
        scr_dir = "/scr-ssd/" + getpass.getuser()
    elif os.path.exists("/scr"):
        scr_dir = "/scr/" + getpass.getuser()
    else:
        scr_dir = "/tmp/scr-" + getpass.getuser()

    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)

    return scr_dir


def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack("I", os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value


def formatted_timestamp(time=None):
    if time is None:
        time = datetime.datetime.now()
    return time.strftime("%d/%m/%Y-%H:%M:%S/%f")


def time_delta_seconds(start, finish=None):
    assert type(start) == str

    t1 = datetime.datetime.strptime(start, "%d/%m/%Y-%H:%M:%S/%f")
    if finish is not None:
        assert type(finish) == str
        t2 = datetime.datetime.strptime(finish, "%d/%m/%Y-%H:%M:%S/%f")
    else:
        t2 = datetime.datetime.now()

    return (t2 - t1).total_seconds()


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


def safe_backward(loss, parameters, accumulate=1, allow_unused=False):
    parameters = list(parameters)  # Capture the generator output
    grads = torch.autograd.grad(loss, parameters, allow_unused=allow_unused)
    nan, inf = False, False
    for g in grads:
        if g is not None:
            nan |= g.isnan().any().item()
            inf |= g.isinf().any().item()

    if not (nan or inf):
        for p, g in zip(parameters, grads):
            if g is None:
                continue

            if p.grad is None:
                p.grad = g / accumulate
            else:
                p.grad += g / accumulate
    else:
        LOG.info(f"Skipping grad accumulation because inf: {inf} nan: {nan}")


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits

def add_sep(tokenizer, model):
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.lm_head.weight.data[-1, :] = model.lm_head.weight.data.mean(0)

def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)

def set_dropout(model, p):
    if p is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = p
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = p
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={p}")


def load_archive(path):
    import torch

    if not os.path.exists(path):
        # We've not passed an explicit path, but a part of the filename
        directories = ["outputs", "multirun"]
        matches = []
        for d in directories:
            search = os.path.join(wd, d)
            for run_dir in os.listdir(search):
                if path in run_dir:
                    matches.append(os.path.join(search, run_dir))
        assert len(matches) == 1, f">1 matches for search {path}; specify exact path"

        full_run_dir = matches[0]
        if "0" in os.listdir(full_run_dir):
            full_run_dir = os.path.join(full_run_dir, "0")
        models_dir = os.path.join(full_run_dir, "models")
        models = os.listdir(models_dir)
        non_bk = [m for m in models if not m.endswith(".bk")]
        assert (
            len(non_bk) == 1
        ), f"Expected a single model in {models_dir}, got {len(non_bk)}"
        path = os.path.join(models_dir, non_bk[0])

    LOG.info(f"Loading checkpoint from {path}")
    archive = torch.load(path, map_location="cpu")
    LOG.info("Load complete.")

    return archive, path


def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output


class EarlyStopper:
    def __init__(self, patience: int, key: str):
        self.best_value = -1 if 'acc' in key else 1e9
        self.best_iter = 0
        self.current_iter = 0
        self.key = key
        self.patience = patience
        self._stop = False

    def update(self, idx, stats):
        assert self.key in stats, f"'{self.key}' not in stats dict"
        value = stats[self.key]
        if 'acc' in self.key:
            new_best = value > self.best_value
        else:
            new_best = value < self.best_value
        if new_best:
            self.best_value = value
            self.best_iter = idx

        self.current_iter = idx
        return new_best

    def should_stop(self):
        self._stop |= self.current_iter - self.best_iter >= self.patience
        return self._stop


class RunningStatAverager:
    def __init__(self, suffix="", exclude=["grad/"], compute_ppl: bool = True):
        self.underlying = None
        self.suffix = suffix
        self.exclude = exclude
        self.compute_ppl = compute_ppl

        self.reset()

    def add(self, d: dict):
        for k, v in d.items():
            if not any([k.startswith(prefix) for prefix in self.exclude]):
                if len(self.suffix):
                    self.underlying[f"{k}_{self.suffix}"].append(v)
                else:
                    self.underlying[k].append(v)

    def average(self):
        average = {}
        for k, v in self.underlying.items():
            if not k.startswith("nll/"):
                average[k] = sum(v) / len(v)
            else:
                assert len(k.split("/")) == 2, f"Invalid key {k}"
                name = k.split("/")[1]
                token_counts = self.underlying[f"n_tokens/{name}"]
                total_nll = sum([nll * c for nll, c in zip(v, token_counts)])
                average[k] = total_nll / sum(token_counts)
                if self.compute_ppl:
                    average[f"perplexity/{name}"] = math.e ** average[k]

        return {
            k: v if not isinstance(v, torch.Tensor) else v.item()
            for k, v in average.items()
        }

    def reset(self):
        self.underlying = defaultdict(list)


class EditBatchSampler:
    def __init__(self, n, n_edits=1, memorize_mode=False, loc_disjoint=True, seed=0):
        self.memorize_mode = memorize_mode
        self.n = n
        self.n_edits = n_edits
        self.loc_disjoint = loc_disjoint
        self.rng = np.random.default_rng(seed)
        self._init()

    def _init(self):
        self.perm = self.rng.permutation(self.n)
        self.edit_position = 0

    def sample(self, batch_size):
        assert (
            batch_size > self.n_edits
        ), "Batch size is interpreted such that batch_size = n_edits + n_loc"

        if self.memorize_mode:
            return list(range(self.n_edits)), list(range(batch_size - self.n_edits))

        if self.edit_position >= self.n:
            self._init()

        edit_idxs = self.perm[self.edit_position : self.edit_position + self.n_edits]
        self.edit_position += self.n_edits

        loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)
        if self.loc_disjoint:
            while len(np.intersect1d(edit_idxs, loc_idxs)) > 0:
                loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)

        return edit_idxs.tolist(), loc_idxs.tolist()


def parent_module(model, pname):
    comps = pname.split(".")
    parent = model
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent


if __name__ == "__main__":
    import random

    stopper = EarlyStopper(1000, "loss/edit")

    data = [
        (100 * idx, {"loss/edit": 2 ** (1 - idx / 10) + random.random()})
        for idx in range(100)
    ]

    for d in data:
        stopper.update(*d)
        print(
            stopper.current_iter,
            stopper.should_stop(),
            stopper.best_iter,
            d[1]["loss/edit"],
        )
