from collections.abc import Sequence
from datetime import UTC, datetime

from ml.purification.schemas import (
    PurificationConfig,
    QuarantineAction,
    QuarantineEvent,
    SampleState,
)
from ml.scoring.risk_fusion import RiskScoreItem


class QuarantineManager:
    """
    Manages the lifecycle state of samples during anomaly screening and purification.
    Maintains an auditable, append-only history of quarantine and restoration events.
    """

    def __init__(self) -> None:
        self._states: dict[str, SampleState] = {}
        self._event_log: list[QuarantineEvent] = []

    def get_state(self, sample_id: str) -> SampleState:
        return self._states.get(sample_id, SampleState.ACTIVE)

    def get_events(self, sample_id: str | None = None) -> list[QuarantineEvent]:
        if sample_id is None:
            return list(self._event_log)
        return [e for e in self._event_log if e.sample_id == sample_id]

    def quarantine_by_risk(
        self,
        risk_items: Sequence[RiskScoreItem],
        config: PurificationConfig | None = None,
    ) -> list[QuarantineEvent]:
        """
        Evaluates risk items and automatically transitions high-risk samples to QUARANTINED.
        """
        cfg = config or PurificationConfig()
        new_events: list[QuarantineEvent] = []

        for item in risk_items:
            current_state = self.get_state(item.sample_id)
            # Check threshold and risk level rules
            should_quarantine = item.risk_score >= cfg.risk_threshold
            if cfg.quarantine_high_risk_only:
                should_quarantine = should_quarantine or (item.risk_level == "HIGH")

            if should_quarantine and current_state != SampleState.QUARANTINED:
                event = QuarantineEvent(
                    sample_id=item.sample_id,
                    action=QuarantineAction.QUARANTINE_AUTO_THRESHOLD,
                    previous_state=current_state,
                    new_state=SampleState.QUARANTINED,
                    reason=f"Risk score {item.risk_score:.4f} exceeded threshold {cfg.risk_threshold:.2f} ({item.risk_level} risk)",
                    threshold=cfg.risk_threshold,
                    timestamp=datetime.now(UTC),
                    metadata={"dominant_evidence": item.dominant_evidence, "detector_score": item.detector_score},
                )
                self._states[item.sample_id] = SampleState.QUARANTINED
                self._event_log.append(event)
                new_events.append(event)

        return new_events

    def quarantine_sample(self, sample_id: str, reason: str = "Manual reviewer quarantine") -> QuarantineEvent:
        """
        Manually quarantines a sample.
        """
        current_state = self.get_state(sample_id)
        event = QuarantineEvent(
            sample_id=sample_id,
            action=QuarantineAction.QUARANTINE_MANUAL,
            previous_state=current_state,
            new_state=SampleState.QUARANTINED,
            reason=reason,
            timestamp=datetime.now(UTC),
        )
        self._states[sample_id] = SampleState.QUARANTINED
        self._event_log.append(event)
        return event

    def restore_sample(self, sample_id: str, reason: str = "Manual reviewer false-positive override") -> QuarantineEvent:
        """
        Restores a previously quarantined sample to RESTORED state.
        """
        current_state = self.get_state(sample_id)
        event = QuarantineEvent(
            sample_id=sample_id,
            action=QuarantineAction.RESTORE_MANUAL,
            previous_state=current_state,
            new_state=SampleState.RESTORED,
            reason=reason,
            timestamp=datetime.now(UTC),
        )
        self._states[sample_id] = SampleState.RESTORED
        self._event_log.append(event)
        return event
