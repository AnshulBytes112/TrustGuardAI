from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.dataset import DatasetModel
from backend.schemas.purification import PurificationRequest, PurificationResponse
from backend.services.purification_service import PurificationService

router = APIRouter(prefix="/purification", tags=["purification"])


class PurificationPreviewResponse(BaseModel):
    dataset_id: str
    total_original_samples: int
    projected_active_count: int
    projected_quarantined_count: int
    projected_restored_count: int
    risk_threshold: float


@router.get("/preview", response_model=PurificationPreviewResponse)
def preview_purification(
    dataset_id: str = Query(..., description="Target dataset ID"),
    experiment_id: str | None = Query(None, description="Optional experiment ID"),
    risk_threshold: float = Query(0.70, description="Risk exclusion threshold"),
    db: Session = Depends(get_db),
):
    """Returns exact live calculation of active vs quarantined samples based on database scores."""
    try:
        data = PurificationService.preview_purification(
            db=db,
            dataset_id=dataset_id,
            experiment_id=experiment_id,
            risk_threshold=risk_threshold,
        )
        return PurificationPreviewResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("", response_model=PurificationResponse)
def trigger_purification(req: PurificationRequest, db: Session = Depends(get_db)):
    """Executes dataset purification based on risk threshold and generates a new clean dataset."""
    try:
        response = PurificationService.purify_dataset(
            db=db,
            dataset_id=req.dataset_id,
            experiment_id=req.experiment_id,
            risk_threshold=req.risk_threshold,
            version_suffix=req.version_suffix,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}", response_model=PurificationResponse)
def get_purification_info(dataset_id: str, db: Session = Depends(get_db)):
    """Gets purification summary for a purified dataset."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    return PurificationResponse(
        original_dataset_id=dataset.id,
        original_dataset_version=dataset.version,
        purified_dataset_id=dataset.id,
        purified_dataset_version=dataset.version,
        total_original_samples=dataset.total_samples,
        active_count=dataset.total_samples,
        quarantined_count=0,
        restored_count=0,
        artifact_uri=dataset.artifact_uri or "",
        created_at=dataset.created_at,
    )
