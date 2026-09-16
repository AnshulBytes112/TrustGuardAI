from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.schemas.retraining import RetrainingRequest, RetrainingResponse
from backend.services.retraining_service import RetrainingService

router = APIRouter(prefix="/retraining", tags=["retraining"])


@router.post("", response_model=RetrainingResponse)
def trigger_retraining(req: RetrainingRequest, db: Session = Depends(get_db)):
    """Runs a downstream retraining benchmark comparing raw vs purified dataset."""
    try:
        response = RetrainingService.run_benchmark(
            db=db,
            raw_dataset_id=req.raw_dataset_id,
            purified_dataset_id=req.purified_dataset_id,
            target_label=req.target_label,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
