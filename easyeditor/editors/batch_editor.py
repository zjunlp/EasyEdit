from enum import Enum


class BatchEditor(Enum):
    CALINET = 'CALINET'
    SERAC = 'SERAC'
    KE = 'KE'
    MEND = 'MEND'
    MEMIT = 'MEMIT'
    PMET = 'PMET'
    FT = 'FT'
    QLoRA = 'QLoRA'
    LoRA = 'LoRA'
    EMMET = "EMMET"

    @staticmethod
    def is_batchable_method(alg_name: str):
        return alg_name == BatchEditor.CALINET.value \
            or alg_name == BatchEditor.SERAC.value \
            or alg_name == BatchEditor.KE.value \
            or alg_name == BatchEditor.MEND.value \
            or alg_name == BatchEditor.MEMIT.value \
            or alg_name == BatchEditor.PMET.value \
            or alg_name == BatchEditor.FT.value \
            or alg_name == BatchEditor.QLoRA.value \
            or alg_name == BatchEditor.LoRA.value \
            or alg_name == BatchEditor.EMMET.value