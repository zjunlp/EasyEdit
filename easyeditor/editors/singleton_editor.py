from enum import Enum

class SingletonEditor(Enum):

    ROME = 'ROME',
    KN = 'KN'

    @staticmethod
    def is_singleton_method(alg_name: str):
        return alg_name == SingletonEditor.ROME.value \
            or alg_name == SingletonEditor.KN.value