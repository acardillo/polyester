from typing import Any
from pydantic import ConfigDict, BaseModel, Field, model_validator

class Relationship(BaseModel):
    """
    Represents a directed connection between two documents.
    
    Example:
        Relationship(
            source_id="stdlib.json.load",
            target_id="stdlib.io.TextIOWrapper",
            relationship_type="parameter_type",
            metadata={"param_name": "fp", "param_position": 0}
        )
    """

    source_id: str
    target_id: str
    relationship_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode='after')
    def validate_no_self_loop(self) -> 'Relationship':
        if self.source_id == self.target_id:
            raise ValueError("Source and target IDs cannot be the same (self-loop detected)")
        return self