from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.core.database import Base


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    detector = Column(String, default="FLARE", nullable=False)
    model_version = Column(String, default="distilbert-base-uncased", nullable=False)
    configuration_json = Column(Text, nullable=True)
    seed = Column(Integer, default=42)
    status = Column(String, default="PENDING", nullable=False)  # PENDING, RUNNING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    threshold = Column(Float, nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    dataset = relationship("DatasetModel", back_populates="experiments")
    scores = relationship("SampleScoreModel", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("MetricModel", back_populates="experiment", cascade="all, delete-orphan")
    quarantine_events = relationship("QuarantineEventModel", back_populates="experiment", cascade="all, delete-orphan")


class SampleScoreModel(Base):
    __tablename__ = "sample_scores"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    sample_id = Column(String, ForeignKey("samples.id"), nullable=False, index=True)
    detector = Column(String, nullable=False)
    raw_score = Column(Float, nullable=False)
    normalized_score = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)  # LOW, MEDIUM, HIGH
    dominant_evidence = Column(String, default="detector")
    evidence_json = Column(Text, nullable=True)

    experiment = relationship("ExperimentModel", back_populates="scores")
    sample = relationship("SampleModel", back_populates="scores")


class MetricModel(Base):
    __tablename__ = "metrics"

    id = Column(String, primary_key=True, index=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)

    experiment = relationship("ExperimentModel", back_populates="metrics")
