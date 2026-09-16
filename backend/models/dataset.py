from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.core.database import Base


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    version = Column(String, default="v1", nullable=False)
    modality = Column(String, default="TEXT", nullable=False)
    label_mode = Column(String, default="FULLY_LABELLED", nullable=False)
    source = Column(String, default="User Upload")
    artifact_uri = Column(String, nullable=True)
    total_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    samples = relationship("SampleModel", back_populates="dataset", cascade="all, delete-orphan")
    experiments = relationship("ExperimentModel", back_populates="dataset", cascade="all, delete-orphan")


class SampleModel(Base):
    __tablename__ = "samples"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False, index=True)
    external_sample_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    text_hash = Column(String, nullable=False, index=True)
    label = Column(String, nullable=True)
    label_status = Column(String, default="KNOWN", nullable=False)
    split = Column(String, default="TRAIN", nullable=False)
    state = Column(String, default="ACTIVE", nullable=False)  # ACTIVE, QUARANTINED, RESTORED
    poison_ground_truth = Column(Integer, nullable=True)  # 1 for True, 0 for False, None for Unknown

    dataset = relationship("DatasetModel", back_populates="samples")
    scores = relationship("SampleScoreModel", back_populates="sample", cascade="all, delete-orphan")
    quarantine_events = relationship("QuarantineEventModel", back_populates="sample", cascade="all, delete-orphan")
