from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.dataset import DatasetModel, SampleModel
from backend.schemas.dataset import (
    DatasetDetailResponse,
    DatasetResponse,
    SampleResponse,
)
from backend.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Uploads and persists a JSONL dataset."""
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl format is supported.")

    content = await file.read()
    ds_name = name or file.filename.replace(".jsonl", "")

    try:
        dataset = DatasetService.create_from_jsonl_content(db, name=ds_name, content=content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Calculate split counts
    train_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "TRAIN").count()
    val_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "VALIDATION").count()
    test_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "TEST").count()

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        version=dataset.version,
        modality=dataset.modality,
        label_mode=dataset.label_mode,
        source=dataset.source,
        total_samples=dataset.total_samples,
        train_count=train_c,
        val_count=val_c,
        test_count=test_c,
        created_at=dataset.created_at,
    )


@router.get("", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """Lists all stored datasets."""
    datasets = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()
    results = []
    for d in datasets:
        train_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "TRAIN").count()
        val_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "VALIDATION").count()
        test_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "TEST").count()
        results.append(
            DatasetResponse(
                id=d.id,
                name=d.name,
                version=d.version,
                modality=d.modality,
                label_mode=d.label_mode,
                source=d.source,
                total_samples=d.total_samples,
                train_count=train_c,
                val_count=val_c,
                test_count=test_c,
                created_at=d.created_at,
            )
        )
    return results


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Gets dataset details and sample overview."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    samples = db.query(SampleModel).filter(SampleModel.dataset_id == dataset_id).limit(100).all()
    train_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "TRAIN").count()
    val_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "VALIDATION").count()
    test_c = db.query(SampleModel).filter(SampleModel.dataset_id == dataset.id, SampleModel.split == "TEST").count()

    sample_items = [
        SampleResponse(
            id=s.id,
            dataset_id=s.dataset_id,
            external_sample_id=s.external_sample_id,
            text=s.text,
            label=s.label,
            label_status=s.label_status,
            split=s.split,
            state=s.state,
        )
        for s in samples
    ]

    return DatasetDetailResponse(
        id=dataset.id,
        name=dataset.name,
        version=dataset.version,
        modality=dataset.modality,
        label_mode=dataset.label_mode,
        source=dataset.source,
        total_samples=dataset.total_samples,
        train_count=train_c,
        val_count=val_c,
        test_count=test_c,
        created_at=dataset.created_at,
        samples=sample_items,
    )


@router.get("/{dataset_id}/samples", response_model=list[SampleResponse])
def get_dataset_samples(
    dataset_id: str,
    split: str | None = None,
    state: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieves paginated samples for a dataset."""
    query = db.query(SampleModel).filter(SampleModel.dataset_id == dataset_id)
    if split:
        query = query.filter(SampleModel.split == split.upper())
    if state:
        query = query.filter(SampleModel.state == state.upper())

    samples = query.offset(offset).limit(limit).all()
    return [
        SampleResponse(
            id=s.id,
            dataset_id=s.dataset_id,
            external_sample_id=s.external_sample_id,
            text=s.text,
            label=s.label,
            label_status=s.label_status,
            split=s.split,
            state=s.state,
        )
        for s in samples
    ]
