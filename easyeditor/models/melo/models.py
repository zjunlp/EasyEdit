import transformers
import torch
import torch.nn as nn
import re
import logging
# from torch.nn import FixableDropout
from .util import scr


LOG = logging.getLogger(__name__)


class CastModule(nn.Module):
    def __init__(self, module: nn.Module, in_cast: torch.dtype = torch.float32, out_cast: torch.dtype = None):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        if model_name.startswith("bert"):
            LOG.info(f"Loading model class {model_name}, cache dir {scr()}")
            self.model = transformers.BertModel.from_pretrained(model_name, cache_dir=scr())
        else:
            self.model = transformers.AutoModel.from_pretrained(model_name, cache_dir=scr())
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        model_output = self.model(*args, **filtered_kwargs)
        if "pooler_output" in model_output.keys():
            pred = self.classifier(model_output.pooler_output)
        else:
            pred = self.classifier(model_output.last_hidden_state[:, 0])

        if "output_hidden_states" in kwargs and kwargs["output_hidden_states"]:
            last_hidden_state = model_output.last_hidden_state
            return pred, last_hidden_state
        else:
            return pred



def get_hf_model(config):
    ModelClass = getattr(transformers, config.model.class_name)
    LOG.info(f"Loading model class {ModelClass} with name {config.model.name} from cache dir {scr()}")
    if config.model.pt is None:
        model = ModelClass.from_pretrained(config.model.name, cache_dir=scr())
    elif config.re_init_model:
        print("Downloading untrained model.")
        model = ModelClass.from_pretrained(config.model.name)
    else:
        try:
            # try to load specified model from local dir
            model = ModelClass.from_pretrained(config.model.pt)
            print(f"Loaded model: {config.model.pt}")
        except:
            print("Couldn't load model: {config.model.pt}. Downloading new model.")
            model = ModelClass.from_pretrained(config.model.name, cache_dir=scr())


    if config.dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = config.dropout
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = config.dropout
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={config.dropout}")
    return model



def get_tokenizer(config):
    tok_name = config.model.tokenizer_name if config.model.tokenizer_name is not None else config.model.name
    tokenizer = getattr(transformers, config.model.tokenizer_class).from_pretrained(tok_name, cache_dir=scr())
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_processor(config):
    processor_name = config.model.processor_name if config.model.processor_name is not None else config.model.name
    processor = getattr(transformers, config.model.processor_class).from_pretrained(processor_name, cache_dir = scr())
    return processor


if __name__ == '__main__':
    m = BertClassifier("bert-base-uncased")
    m(torch.arange(5)[None, :])
    import pdb; pdb.set_trace()
