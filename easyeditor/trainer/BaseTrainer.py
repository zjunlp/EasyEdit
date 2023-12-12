import json
import logging
import os
import shutil
import tempfile
import time

import torch
import copy
from .losses import kl_loc_loss
from .utils import *
from omegaconf import OmegaConf
from .models import *
from torch.utils.data import Dataset, DataLoader
from ..util.alg_train_dict import ALG_TRAIN_DICT
import importlib
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        LOG.info(f'Config: {config}')
        model_ = get_model(config)
        self.alg_module = ALG_TRAIN_DICT[config.alg.upper()]
        LOG.info(f"Loading class {config.alg.upper()} from module {self.alg_module}")
        self.model = self.alg_module(model_, config, lambda: copy.deepcopy(model_))

        self.config = config

        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = self.model.model

        if self.config.model_parallel:
            self.config.device = self.model.model.device
        if not self.config.model_parallel and hasattr(self.config, 'device'):
            self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        if 'minigpt4' in self.config.model_name.lower() or 'blip2' in self.config.model_name.lower():
            collate_fn = train_set.collate_fn
        elif 't5' in self.config.model_class.lower():
            collate_fn = train_set.collate_fn
        elif 'gpt' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'llama' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'automodel' in self.config.model_class.lower():
            collate_fn = train_set.collate_gpt_fn
        elif 'qwen' in self.config.model_name.lower():
            collate_fn = train_set.collate_gpt_fn
        else:
            raise NotImplementedError(f'Model {self.config.model_class} not supported yet.')

        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size,
                                       shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=self.config.val_batch_size,
                                       shuffle=False, collate_fn=collate_fn)

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only:
            self.OptimizerClass = getattr(torch.optim, config.opt)
            LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
            self.opt = self.OptimizerClass(self.model.outer_parameters(), lr=config.lr)

        if config.archive is not None:
            archive, config.archive = load_archive(str(config.archive))
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only:
                self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = (
                archive  # Save for later to load e.g. lr_opt params if they exist
            )
        else:
            self.archive = None

        # # outfiles
        # with open(os.getcwd() + "/config.json", "w") as f:
        #     json.dump(OmegaConf.to_container(config), f)

        model_dir = os.path.join(config.results_dir, "models", config.alg)
        if not (self.config.debug and not self.config.save) and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        safe_model_name = self.config.model_name.split("/")[-1]  # Make sure no slashes
        self.save_path = f"{model_dir}/{safe_model_name}"

        self.start_time = formatted_timestamp()

    def save_state(self, stats):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "lr_opt": self.lr_opt.state_dict() if self.lr_opt is not None else None,
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter,
        }
        LOG.info(f"Saving model to {self.save_path}")

        if os.path.exists(self.save_path):
            bk_path = f"{self.save_path}.bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.rename(self.save_path, bk_path)

        torch.save(obj, self.save_path)
        LOG.info("Write complete.")

    def echo(self, train_step, info_dict, pretty=False):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k

            LOG.info(f"Step {train_step}:")
            LOG.info(
                sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()])
            )

    def run(self):
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(
            self.config.early_stop_patience, self.config.early_stop_key
        )
        self.global_iter = 0

        assert self.config.max_epochs is not None or self.config.max_iters is not None
        if self.config.max_epochs is not None:
            if self.config.max_iters is not None:
                self.config.max_iters = min(self.config.max_iters, self.config.max_epochs * len(self.train_set))
            else:
                self.config.max_iters = self.config.max_epochs * len(self.train_set)
            LOG.info(f'MAX EPOCH: {self.config.max_epochs}, set max iters to {self.config.max_iters}')

        self.epoches = round(float(self.config.max_iters) / (len(self.train_set) / self.config.batch_size))
        self.global_iter = 0
        for epoch in range(self.epoches):
            for i, batch in enumerate(self.train_loader):
                self.global_iter += 1
                if self.global_iter >= self.config.max_iters:
                    break
                if not self.config.eval_only:
                    train_info = self.train_step(batch)
                    averager.add(train_info)

                    if self.global_iter % self.config.log_interval == 0:
                        avg_info = averager.average()
                        averager.reset()
                        self.echo(self.global_iter, avg_info)
                if self.global_iter % self.config.val_interval == 0:
                    val_info = self.validate(steps=self.config.val_steps)
                    self.echo(self.global_iter, val_info)
                    if stopper.update(self.global_iter, val_info):
                        self.save_state(val_info)  # New best
                    if stopper.should_stop():
                        LOG.info(
                            f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps"
                        )
                        break

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter+1} steps.")

        if not self.config.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                archive = torch.load(self.save_path, map_location="cpu")
                LOG.info(
                    f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}"
                )
                self.model.to("cpu")
                self.model.load_state_dict(archive["model"])
                self.model.to(self.config.device)

        val_steps = self.config.val_steps if self.config.debug else None
        val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)

        if self.config.results_dir is not None:
            results_path = f"{self.config.results_dir}/results.json"
        else:
            results_path = f"{os.getcwd()}/results.json"

        with open(results_path, "w") as f:
            json.dump(
                {"results": val_info}, f
            )
            LOG.info("Wrote results to:")
            LOG.info(results_path)
