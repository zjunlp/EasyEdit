import copy
import importlib
import logging
import random

import hydra
import models
import numpy as np
import torch
import utils
from omegaconf import OmegaConf
from trainer import EditTrainer

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(
    format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


@hydra.main(config_path="config", config_name="config")
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    if config.task == "gen" or config.task == "wiki":
        add_padding(tokenizer, model)
        from data_classes.wiki import GenDataset

        train_set = GenDataset("train", tokenizer, config, config.data.path, pct=10)
        val_set = GenDataset("validation", tokenizer, config, config.data.path, pct=10)
    elif config.task == "fc" or config.task == "fever":
        from data_classes.fever import BinaryAugmentedKILT

        train_set = BinaryAugmentedKILT(
            tokenizer, f"{base_dir}/data/fever/fever-train-kilt.jsonl", config
        )
        val_set = BinaryAugmentedKILT(
            tokenizer, f"{base_dir}/data/fever/fever-dev-kilt.jsonl", config
        )
    elif config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import Seq2SeqAugmentedKILT

        train_set = Seq2SeqAugmentedKILT(
            tokenizer,
            f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl",
            config,
        )
        val_set = Seq2SeqAugmentedKILT(
            tokenizer,
            f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl",
            config,
        )
    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(
                config.ft.locality.batch_size + 1
            )
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(
                train_set.edit_generator(config.ft.locality.batch_size + 1)
            )["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    trainer = EditTrainer(alg, config, train_set, val_set)
    trainer.run()


if __name__ == "__main__":
    run()
