import numpy as np
from pydantic import BaseModel, Field


class RepresentationResult(BaseModel):
    sample_ids: list[str]
    representations: np.ndarray
    layer_representations: dict[int, np.ndarray] | None = None
    model_name: str = Field(..., min_length=1)
    max_length: int = Field(..., gt=0)

    class Config:
        arbitrary_types_allowed = True
