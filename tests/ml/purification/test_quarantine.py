from ml.purification.quarantine import QuarantineManager
from ml.purification.schemas import PurificationConfig, QuarantineAction, SampleState
from ml.scoring.risk_fusion import RiskScoreItem


def test_quarantine_manager_manual_workflow():
    qm = QuarantineManager()

    assert qm.get_state("s1") == SampleState.ACTIVE

    # Manual quarantine
    q_event = qm.quarantine_sample("s1", reason="Suspected trigger token")
    assert q_event.action == QuarantineAction.QUARANTINE_MANUAL
    assert q_event.new_state == SampleState.QUARANTINED
    assert qm.get_state("s1") == SampleState.QUARANTINED

    # Manual restore
    r_event = qm.restore_sample("s1", reason="Verified clean by expert")
    assert r_event.action == QuarantineAction.RESTORE_MANUAL
    assert r_event.new_state == SampleState.RESTORED
    assert qm.get_state("s1") == SampleState.RESTORED

    # Verify event audit trail
    events = qm.get_events("s1")
    assert len(events) == 2


def test_quarantine_by_risk_threshold():
    qm = QuarantineManager()
    config = PurificationConfig(risk_threshold=0.70, quarantine_high_risk_only=True)

    items = [
        RiskScoreItem(
            sample_id="clean_1",
            risk_score=0.15,
            risk_level="LOW",
            detector_score=0.1,
            cluster_score=0.1,
            layer_score=0.1,
            dominant_evidence="detector",
        ),
        RiskScoreItem(
            sample_id="poison_1",
            risk_score=0.85,
            risk_level="HIGH",
            detector_score=0.9,
            cluster_score=0.8,
            layer_score=0.8,
            dominant_evidence="detector",
        ),
    ]

    events = qm.quarantine_by_risk(items, config)

    assert len(events) == 1
    assert events[0].sample_id == "poison_1"
    assert qm.get_state("clean_1") == SampleState.ACTIVE
    assert qm.get_state("poison_1") == SampleState.QUARANTINED
