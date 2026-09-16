from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from backend.core.database import Base


class QuarantineEventModel(Base):
    __tablename__ = "quarantine_events"

    id = Column(String, primary_key=True, index=True)
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=True, index=True)
    action = Column(String, nullable=False)  # QUARANTINE_AUTO_THRESHOLD, QUARANTINE_MANUAL, RESTORE_MANUAL
    previous_state = Column(String, default="ACTIVE")
    new_state = Column(String, default="QUARANTINED")
    reason = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    sample = relationship("SampleModel", back_populates="quarantine_events")
    experiment = relationship("ExperimentModel", back_populates="quarantine_events")
