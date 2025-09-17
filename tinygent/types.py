from pydantic import BaseModel


class TinyModel(BaseModel):
    def __hash__(self) -> int:
        return hash(self.model_dump_json())
