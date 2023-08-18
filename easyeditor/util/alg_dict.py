from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.serac import SERACHparams, SeracRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset
from ..models.ike import IKEHyperParams, apply_ike_to_model


ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    'KN': apply_kn_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset
}