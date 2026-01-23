from pydantic import BaseModel


def _get_doc(doc: str | None) -> str:
    return doc or 'Description not provided'


class TinyModel(BaseModel):
    def __hash__(self) -> int:
        return hash(self.model_dump_json())

    @property
    def doc(self) -> str:
        return _get_doc(self.__doc__)

    @classmethod
    def doc_cls(cls) -> str:
        return _get_doc(cls.__doc__)
