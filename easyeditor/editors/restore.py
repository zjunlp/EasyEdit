from ..util import nethook
from ..util.device import copy_to_param
import torch


CALLABLE_RESTORE_ALGS = {"KN", "GRACE", "WISE"}
PEFT_RESTORE_ALGS = {"LoRA", "QLoRA", "DPO"}
KEEP_EDITED_MODEL_ALGS = {"MELO"}


def restore_after_edit(editor, edited_model, weights_copy):
    """Restore editor.model after a non-sequential edit.

    EasyEdit algorithms currently return different restore handles:
    a parameter snapshot dict, a callable reset function, or a PEFT wrapper.
    Keep those existing contracts in one place so editor loops do not need
    algorithm-specific cleanup branches.
    """
    alg_name = getattr(editor, "alg_name", None)

    if alg_name in CALLABLE_RESTORE_ALGS:
        if weights_copy is not None:
            with torch.no_grad():
                weights_copy()
        return edited_model

    if alg_name in PEFT_RESTORE_ALGS:
        restored_model = edited_model.unload()
        if restored_model is not None:
            editor.model = restored_model
        if hasattr(editor.model, "peft_config"):
            del editor.model.peft_config
        return edited_model

    if alg_name in KEEP_EDITED_MODEL_ALGS:
        editor.model = edited_model
        return edited_model

    if weights_copy is not None:
        with torch.no_grad():
            for key, value in weights_copy.items():
                copy_to_param(nethook.get_parameter(editor.model, key), value)

    return edited_model
