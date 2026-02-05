from typing import Any, Optional
from pydantic import ConfigDict, BaseModel, Field, field_validator
from .relationship import Relationship

class Document(BaseModel):
    """
    Universal representation of a retrievable information unit.
    
    Example:
        Document(
            id="stdlib.json.load",
            content="Deserialize fp (a .read()-supporting file-like object containing a JSON document) to a Python object using this conversion table",
            metadata={
                "module": "json",
                "function": "load",
                "signature": "load(fp, *, cls=None, object_hook=None, ...)",
                "python_version": "3.11"
            },
            embedding=[0.123, -0.456, 0.789, ...],  # 384 dimensions
            relationships=[
                Relationship(source_id="stdlib.json.load", target_id="stdlib.io.TextIOWrapper", ...)
            ]
        )
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    relationships: list[Relationship] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator('embedding')
    def validate_embedding(cls, v: Optional[list[float]]):
        if v is not None:
            if len(v) == 0:
                raise ValueError("Embedding cannot be an empty list")
            if not all(isinstance(elem, float) for elem in v):
                raise ValueError("Embedding elements must be float")
        return v
    
