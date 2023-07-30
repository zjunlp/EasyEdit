from enum import Enum

class BatchEditor(Enum):

    CALINET = 'CALINET'
    SERAC = 'SERAC'
    KE = 'KE'
    MEND = 'MEND'
    MEMIT = 'MEMIT'
    FT = 'FT'

    @staticmethod
    def is_batchable_method(alg_name: str):
        return alg_name == BatchEditor.CALINET.value \
            or alg_name == BatchEditor.SERAC.value \
            or alg_name == BatchEditor.KE.value \
            or alg_name == BatchEditor.MEND.value \
            or alg_name == BatchEditor.MEMIT.value \
            or alg_name == BatchEditor.FT.value
