from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.dataset import DatasetModel, SampleModel
from backend.models.experiment import ExperimentModel, MetricModel

router = APIRouter(prefix="/stats", tags=["stats"])


class OverviewStats(BaseModel):
    total_datasets: int
    total_samples: int
    total_quarantined: int
    total_restored: int
    total_scans: int
    completed_scans: int
    running_scans: int
    avg_auroc: float | None
    avg_precision: float | None
    avg_recall: float | None
    avg_f1: float | None
    recent_scans: list[dict[str, Any]]
    recent_datasets: list[dict[str, Any]]


@router.get("/overview", response_model=OverviewStats)
def get_overview_stats(db: Session = Depends(get_db)):
    """Computes real-time aggregated security stats directly from the database."""
    total_datasets = db.query(func.count(DatasetModel.id)).scalar() or 0
    total_samples = db.query(func.count(SampleModel.id)).scalar() or 0
    total_quarantined = db.query(func.count(SampleModel.id)).filter(SampleModel.state == "QUARANTINED").scalar() or 0
    total_restored = db.query(func.count(SampleModel.id)).filter(SampleModel.state == "RESTORED").scalar() or 0

    total_scans = db.query(func.count(ExperimentModel.id)).scalar() or 0
    completed_scans = db.query(func.count(ExperimentModel.id)).filter(ExperimentModel.status == "COMPLETED").scalar() or 0
    running_scans = db.query(func.count(ExperimentModel.id)).filter(ExperimentModel.status.in_(["RUNNING", "PENDING"])).scalar() or 0

    # Aggregate metric averages across completed scans
    avg_auroc = db.query(func.avg(MetricModel.metric_value)).filter(MetricModel.metric_name == "auroc").scalar()
    avg_precision = db.query(func.avg(MetricModel.metric_value)).filter(MetricModel.metric_name == "precision").scalar()
    avg_recall = db.query(func.avg(MetricModel.metric_value)).filter(MetricModel.metric_name == "recall").scalar()
    avg_f1 = db.query(func.avg(MetricModel.metric_value)).filter(MetricModel.metric_name == "f1_score").scalar()

    # Recent scans with their metrics
    recent_scans_db = db.query(ExperimentModel).order_by(ExperimentModel.started_at.desc()).limit(5).all()
    recent_scans = []
    for s in recent_scans_db:
        metrics_dict = {m.metric_name: m.metric_value for m in s.metrics}
        recent_scans.append({
            "id": s.id,
            "dataset_id": s.dataset_id,
            "name": s.name,
            "detector": s.detector,
            "status": s.status,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "metrics": metrics_dict,
        })

    # Recent datasets
    recent_datasets_db = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).limit(5).all()
    recent_datasets = []
    for d in recent_datasets_db:
        train_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "TRAIN").count()
        val_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "VALIDATION").count()
        test_c = db.query(SampleModel).filter(SampleModel.dataset_id == d.id, SampleModel.split == "TEST").count()
        recent_datasets.append({
            "id": d.id,
            "name": d.name,
            "version": d.version,
            "modality": d.modality,
            "total_samples": d.total_samples,
            "train_count": train_c,
            "val_count": val_c,
            "test_count": test_c,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })

    return OverviewStats(
        total_datasets=total_datasets,
        total_samples=total_samples,
        total_quarantined=total_quarantined,
        total_restored=total_restored,
        total_scans=total_scans,
        completed_scans=completed_scans,
        running_scans=running_scans,
        avg_auroc=float(avg_auroc) if avg_auroc is not None else None,
        avg_precision=float(avg_precision) if avg_precision is not None else None,
        avg_recall=float(avg_recall) if avg_recall is not None else None,
        avg_f1=float(avg_f1) if avg_f1 is not None else None,
        recent_scans=recent_scans,
        recent_datasets=recent_datasets,
    )
