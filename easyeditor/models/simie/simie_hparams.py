from dataclasses import dataclass, field
from typing import List, Optional, Any
import yaml
import os

from ...util.hparams import HyperParams


@dataclass
class SimIEHyperParams(HyperParams):
    # SimIE specific parameters
    base_method: str  
    lamHyper: float
    init_model: bool
    solver: str  
    fast: bool 
    # base parameters
    alg_name: str 
    model_name: Optional[str]
    device: int 
    max_length: int  
    model_parallel: bool 
    fp16: bool
    _base_hparams: Any = field(default=None, repr=False, init=False)
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
        
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        
        base_method = config['base_method']
        base_hparams_class = cls._get_base_hparams_class(base_method)

        # Determine base method's config path
        base_hparams_path = hparams_name_or_path.replace('/SimIE/', f'/{base_method}/')
        
        print(f"[SimIE] Loading base method ({base_method}) hparams from: {base_hparams_path}")
        base_hparams = base_hparams_class.from_hparams(base_hparams_path)
        simie_hparams = cls(**config)
        simie_hparams._base_hparams = base_hparams
        
        print(f"[SimIE] Loaded SimIE hparams with base method: {base_method}")
        print(f"[SimIE] lamHyper={simie_hparams.lamHyper}, solver={simie_hparams.solver}, init_model={simie_hparams.init_model}")
        
        return simie_hparams
    
    @staticmethod
    def _get_base_hparams_class(base_method: str):
        if base_method == 'ROME':
            from ..rome import ROMEHyperParams
            return ROMEHyperParams
        elif base_method == 'MEMIT':
            from ..memit import MEMITHyperParams
            return MEMITHyperParams
        elif base_method == 'FT':
            from ..ft import FTHyperParams
            return FTHyperParams
        elif base_method == 'MEND':
            from ..mend import MENDHyperParams
            return MENDHyperParams
        elif base_method == 'KN':
            from ..kn import KNHyperParams
            return KNHyperParams
        elif base_method == 'LoRA':
            from ..lora import LoRAHyperParams
            return LoRAHyperParams
        elif base_method == 'GRACE':
            from ..grace import GraceHyperParams
            return GraceHyperParams
        elif base_method == 'WISE':
            from ..wise import WISEHyperParams
            return WISEHyperParams
        else:
            raise ValueError(
                f"Unknown base method: {base_method}. "
                f"Supported methods: ROME, MEMIT, FT, MEND, KN, LoRA, GRACE, WISE"
            )
        
    @property
    def base_hparams(self):
        if self._base_hparams is None:
            raise RuntimeError(
                "Base hparams not loaded. "
                "Please use SimIEHyperParams.from_hparams() to load config."
            )
        return self._base_hparams
    