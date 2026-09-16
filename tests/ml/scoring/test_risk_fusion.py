import pytest

from ml.scoring.risk_fusion import RiskFusionConfig, RiskFusionEngine


def test_risk_fusion_weighted_calculation():
    config = RiskFusionConfig(w_detector=0.5, w_cluster=0.3, w_layer=0.2, low_threshold=0.4, high_threshold=0.7)
    engine = RiskFusionEngine(config)

    # Clean sample: all low scores -> LOW risk
    clean_item = engine.fuse_sample("s_clean", detector_score=0.1, cluster_score=0.1, layer_score=0.1)
    assert clean_item.risk_score == pytest.approx(0.1, rel=1e-3)
    assert clean_item.risk_level == "LOW"

    # Suspicious sample: high detector & cluster -> HIGH risk
    high_item = engine.fuse_sample("s_poison", detector_score=0.9, cluster_score=0.8, layer_score=0.7)
    assert high_item.risk_score == pytest.approx(0.83, rel=1e-2)
    assert high_item.risk_level == "HIGH"
    assert high_item.dominant_evidence == "detector"


def test_risk_fusion_batch():
    engine = RiskFusionEngine()
    result = engine.fuse_batch(
        sample_ids=["s1", "s2", "s3"],
        detector_scores=[0.1, 0.5, 0.9],
        cluster_scores=[0.1, 0.5, 0.9],
        layer_scores=[0.1, 0.5, 0.9],
    )

    assert len(result.items) == 3
    assert result.low_count == 1
    assert result.medium_count == 1
    assert result.high_count == 1


def test_risk_fusion_invalid_weights():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        RiskFusionConfig(w_detector=0.8, w_cluster=0.8, w_layer=0.2)
