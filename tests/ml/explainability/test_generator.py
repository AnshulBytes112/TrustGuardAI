from ml.data.schemas import LabelStatus, Sample, Split
from ml.explainability.generator import ExplanationGenerator
from ml.scoring.layer_scores import LayerAnomalyProfile
from ml.scoring.risk_fusion import RiskScoreItem


def test_token_saliency_generation():
    generator = ExplanationGenerator()
    text = "Please transfer $10,000 immediately to account XYZ-99!"
    attributions = generator.generate_token_saliency(text, risk_score=0.85)

    assert len(attributions) > 0
    tokens = [a.token for a in attributions]
    assert "$10,000" in tokens
    assert any(a.is_suspicious_span for a in attributions)


def test_sample_explanation_synthesis():
    generator = ExplanationGenerator()
    sample = Sample(
        sample_id="sample_001",
        text="This is a test sample with anomalous trigger phrase.",
        label="POSITIVE",
        label_status=LabelStatus.KNOWN,
        split=Split.TEST,
        dataset_id="test_ds",
        dataset_version="v1",
    )

    risk_item = RiskScoreItem(
        sample_id="sample_001",
        risk_score=0.82,
        risk_level="HIGH",
        detector_score=0.88,
        cluster_score=0.75,
        layer_score=0.80,
        dominant_evidence="detector",
    )

    layer_profile = LayerAnomalyProfile(
        sample_id="sample_001",
        layer_scores={1: 0.2, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.8, 6: 0.9},
        layer_attributions={1: 0.06, 2: 0.09, 3: 0.12, 4: 0.19, 5: 0.25, 6: 0.29},
        dominant_layer=6,
        trajectory="late",
    )

    explanation = generator.generate_explanation(sample, risk_item, layer_profile)

    assert explanation.sample_id == "sample_001"
    assert explanation.label == "POSITIVE"
    assert "HIGH" in explanation.evidence_summary
    assert "Layer 6" in explanation.evidence_summary
    assert len(explanation.token_attributions) > 0


def test_unlabelled_sample_explanation():
    generator = ExplanationGenerator()
    sample = Sample(
        sample_id="sample_unlabelled",
        text="Unlabelled text sample.",
        label=None,
        label_status=LabelStatus.UNKNOWN,
        split=Split.TEST,
        dataset_id="test_ds",
        dataset_version="v1",
    )

    risk_item = RiskScoreItem(
        sample_id="sample_unlabelled",
        risk_score=0.20,
        risk_level="LOW",
        detector_score=0.15,
        cluster_score=0.20,
        layer_score=0.25,
        dominant_evidence="layer",
    )

    layer_profile = LayerAnomalyProfile(
        sample_id="sample_unlabelled",
        layer_scores={1: 0.1, 2: 0.1},
        layer_attributions={1: 0.5, 2: 0.5},
        dominant_layer=1,
        trajectory="uniform",
    )

    explanation = generator.generate_explanation(sample, risk_item, layer_profile)

    assert explanation.label is None
    assert explanation.label_status == "UNKNOWN"
    assert "LOW" in explanation.evidence_summary
