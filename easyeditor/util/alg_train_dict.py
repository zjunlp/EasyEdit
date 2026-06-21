from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import MALMEN


# Trainer-backed algorithms. Multimodal edit-only algorithms such as IKE,
# LoRA, WISE, and GRACE are registered in ALG_MULTIMODAL_DICT instead.
MULTIMODAL_TRAIN_ALGS = {
    'MEND',
    'SERAC',
    'SERAC_MULTI',
}

ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'MALMEN': MALMEN,
}
