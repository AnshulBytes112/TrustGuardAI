import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.dataset import SampleModel
from backend.models.experiment import SampleScoreModel
from backend.models.quarantine import QuarantineEventModel
from backend.schemas.sample import (
    QuarantineEventResponse,
    SampleActionRequest,
    SampleDetailResponse,
)

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("/{sample_id}", response_model=SampleDetailResponse)
def get_sample_investigation(sample_id: str, db: Session = Depends(get_db)):
    """Retrieves full XAI explanation, layer decomposition, and history for a sample."""
    sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found.")

    score_model = (
        db.query(SampleScoreModel)
        .filter(SampleScoreModel.sample_id == sample_id)
        .order_by(SampleScoreModel.risk_score.desc())
        .first()
    )

    events = (
        db.query(QuarantineEventModel)
        .filter(QuarantineEventModel.sample_id == sample_id)
        .order_by(QuarantineEventModel.timestamp.desc())
        .all()
    )

    evidence = json.loads(score_model.evidence_json) if (score_model and score_model.evidence_json) else {}
    history = [
        QuarantineEventResponse(
            id=e.id,
            sample_id=e.sample_id,
            action=e.action,
            previous_state=e.previous_state,
            new_state=e.new_state,
            reason=e.reason,
            timestamp=e.timestamp,
        )
        for e in events
    ]

    return SampleDetailResponse(
        id=sample.id,
        dataset_id=sample.dataset_id,
        external_sample_id=sample.external_sample_id,
        text=sample.text,
        label=sample.label,
        label_status=sample.label_status,
        split=sample.split,
        state=sample.state,
        risk_score=score_model.risk_score if score_model else 0.0,
        risk_level=score_model.risk_level if score_model else "LOW",
        dominant_evidence=score_model.dominant_evidence if score_model else "detector",
        token_attributions=evidence.get("token_attributions", []),
        layer_scores={},
        layer_attributions=evidence.get("layer_attributions", {}),
        dominant_layer=evidence.get("dominant_layer", 1),
        trajectory=evidence.get("trajectory", "uniform"),
        evidence_summary=evidence.get("evidence_summary", "No anomalous behavior detected."),
        quarantine_history=history,
    )


@router.post("/{sample_id}/quarantine", response_model=SampleDetailResponse)
def quarantine_sample(sample_id: str, req: SampleActionRequest, db: Session = Depends(get_db)):
    """Manually flags and quarantines a sample."""
    sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found.")

    prev_state = sample.state
    sample.state = "QUARANTINED"

    event = QuarantineEventModel(
        id=f"qe_{uuid.uuid4().hex[:12]}",
        sample_id=sample.id,
        action="QUARANTINE_MANUAL",
        previous_state=prev_state,
        new_state="QUARANTINED",
        reason=req.reason,
        timestamp=datetime.now(UTC),
    )
    db.add(event)
    db.commit()
    db.refresh(sample)

    return get_sample_investigation(sample_id=sample.id, db=db)


@router.post("/{sample_id}/restore", response_model=SampleDetailResponse)
def restore_sample(sample_id: str, req: SampleActionRequest, db: Session = Depends(get_db)):
    """Restores a quarantined sample as verified clean."""
    sample = db.query(SampleModel).filter(SampleModel.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found.")

    prev_state = sample.state
    sample.state = "RESTORED"

    event = QuarantineEventModel(
        id=f"qe_{uuid.uuid4().hex[:12]}",
        sample_id=sample.id,
        action="RESTORE_MANUAL",
        previous_state=prev_state,
        new_state="RESTORED",
        reason=req.reason,
        timestamp=datetime.now(UTC),
    )
    db.add(event)
    db.commit()
    db.refresh(sample)

    return get_sample_investigation(sample_id=sample.id, db=db)
