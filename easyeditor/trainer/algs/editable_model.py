import torch.nn as nn
from copy import deepcopy

from ..losses import masked_log_probs
from ..utils import _logits, shift_targets


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.model = model
        self.config = deepcopy(config)
        self.model_constructor = model_constructor

        def _edit_loss_fn(config, pred, targ):
            if 'minigpt4' in config.model_name.lower() or 'blip' in self.config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 't5' in config.model_class.lower():
                return masked_log_probs(config, pred, targ)
            elif 'gpt' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'llama' in config.model_class.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'internlm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'chatglm' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            elif 'qwen' in config.model_name.lower():
                return masked_log_probs(config, pred, targ, shift=True)
            else:
                return masked_log_probs(config, pred, targ)

        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = masked_log_probs

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass
