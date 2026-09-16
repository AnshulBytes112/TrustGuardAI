import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal, get_db
from backend.models.dataset import SampleModel
from backend.models.experiment import ExperimentModel, MetricModel, SampleScoreModel
from backend.schemas.scan import (
    MetricItemResponse,
    ScanCreateRequest,
    ScanResponse,
    ScanSampleItem,
)
from backend.services.scan_service import ScanService

router = APIRouter(prefix="/scans", tags=["scans"])


def run_scan_in_background(experiment_id: str, dataset_id: str, detector: str, layers: tuple[int, ...], seed: int):
    db = SessionLocal()
    try:
        ScanService.execute_scan(
            db=db,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            detector_type=detector,
            layers=layers,
            seed=seed,
        )
    finally:
        db.close()


@router.post("", response_model=ScanResponse)
def create_scan(
    req: ScanCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Creates a new anomaly detection scan."""
    exp_id = f"exp_{uuid.uuid4().hex[:12]}"
    exp_name = req.name or f"Scan_{req.detector}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    experiment = ExperimentModel(
        id=exp_id,
        dataset_id=req.dataset_id,
        name=exp_name,
        detector=req.detector,
        model_version="distilbert-base-uncased",
        configuration_json=json.dumps({"layers": list(req.layers), "threshold": req.threshold, "seed": req.seed}),
        seed=req.seed,
        status="PENDING",
        threshold=req.threshold,
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    # Launch background scan
    background_tasks.add_task(
        run_scan_in_background,
        experiment_id=exp_id,
        dataset_id=req.dataset_id,
        detector=req.detector,
        layers=req.layers,
        seed=req.seed,
    )

    return ScanResponse(
        id=experiment.id,
        dataset_id=experiment.dataset_id,
        name=experiment.name,
        detector=experiment.detector,
        model_version=experiment.model_version,
        status=experiment.status,
        started_at=experiment.started_at,
        threshold=experiment.threshold,
        metrics={},
    )


@router.get("", response_model=list[ScanResponse])
def list_scans(db: Session = Depends(get_db)):
    """Lists all scans and experiments."""
    experiments = db.query(ExperimentModel).order_by(ExperimentModel.started_at.desc()).all()
    results = []
    for exp in experiments:
        metrics_dict = {m.metric_name: m.metric_value for m in exp.metrics}
        results.append(
            ScanResponse(
                id=exp.id,
                dataset_id=exp.dataset_id,
                name=exp.name,
                detector=exp.detector,
                model_version=exp.model_version,
                status=exp.status,
                error_message=exp.error_message,
                threshold=exp.threshold,
                started_at=exp.started_at,
                completed_at=exp.completed_at,
                metrics=metrics_dict,
            )
        )
    return results


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: str, db: Session = Depends(get_db)):
    """Gets details and status for a specific scan."""
    exp = db.query(ExperimentModel).filter(ExperimentModel.id == scan_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Scan not found.")

    metrics_dict = {m.metric_name: m.metric_value for m in exp.metrics}
    return ScanResponse(
        id=exp.id,
        dataset_id=exp.dataset_id,
        name=exp.name,
        detector=exp.detector,
        model_version=exp.model_version,
        status=exp.status,
        error_message=exp.error_message,
        threshold=exp.threshold,
        started_at=exp.started_at,
        completed_at=exp.completed_at,
        metrics=metrics_dict,
    )


@router.get("/{scan_id}/samples", response_model=list[ScanSampleItem])
def get_scan_samples(
    scan_id: str,
    risk_level: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieves ranked suspicious samples for an experiment."""
    query = (
        db.query(SampleScoreModel, SampleModel)
        .join(SampleModel, SampleScoreModel.sample_id == SampleModel.id)
        .filter(SampleScoreModel.experiment_id == scan_id)
    )

    if risk_level:
        query = query.filter(SampleScoreModel.risk_level == risk_level.upper())

    # Order by risk score descending
    query = query.order_by(SampleScoreModel.risk_score.desc()).offset(offset).limit(limit)
    rows = query.all()

    results = []
    for score_m, sample_m in rows:
        evidence = json.loads(score_m.evidence_json) if score_m.evidence_json else {}
        results.append(
            ScanSampleItem(
                sample_id=sample_m.id,
                external_sample_id=sample_m.external_sample_id,
                text=sample_m.text,
                label=sample_m.label,
                label_status=sample_m.label_status,
                split=sample_m.split,
                state=sample_m.state,
                raw_score=score_m.raw_score,
                normalized_score=score_m.normalized_score,
                risk_score=score_m.risk_score,
                risk_level=score_m.risk_level,
                dominant_evidence=score_m.dominant_evidence,
                evidence=evidence,
            )
        )
    return results


@router.get("/{scan_id}/metrics", response_model=list[MetricItemResponse])
def get_scan_metrics(scan_id: str, db: Session = Depends(get_db)):
    """Retrieves metrics evaluated during a scan."""
    metrics = db.query(MetricModel).filter(MetricModel.experiment_id == scan_id).all()
    return [MetricItemResponse(metric_name=m.metric_name, metric_value=m.metric_value) for m in metrics]
