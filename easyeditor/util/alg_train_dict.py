from ..trainer import EFK
from ..trainer import MEND
from ..trainer import ENN
from ..trainer import SERAC


ALG_TRAIN_DICT = {
    "KE": EFK,
    'ENN': ENN,
    'MEND': MEND,
    'SERAC': SERAC
}