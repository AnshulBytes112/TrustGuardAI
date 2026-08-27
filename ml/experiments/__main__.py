import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ml.experiments.runner import ExperimentRunner
from ml.experiments.schemas import ExperimentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ml.experiments")

def atomic_write(filepath: Path, content: str) -> None:
    """Writes content to filepath atomically using a .tmp file."""
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

def main():
    parser = argparse.ArgumentParser(description="Run a TrustGuard AI anomaly detection experiment.")
    parser.add_argument("config_path", type=str, help="Path to the JSON experiment configuration file.")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = f.read()
            config = ExperimentConfig.model_validate_json(config_json)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to parse or validate configuration: {e}")
        sys.exit(1)

    fingerprint = config.compute_fingerprint()
    artifact_dir = Path(f"artifacts/experiments/{fingerprint}")

    # Check for existing valid artifacts
    result_path = artifact_dir / "result.json"
    if result_path.exists():
        logger.info(f"Artifact already exists for fingerprint {fingerprint}. Verifying...")
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                existing_result = json.load(f)
                
            if existing_result.get("experiment_fingerprint") == fingerprint:
                logger.info("Found existing identical experiment result. Skipping re-execution.")
                sys.exit(0)
            else:
                logger.error("Fingerprint collision or artifact corruption detected! Existing result fingerprint does not match.")
                sys.exit(1)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to read existing artifact: {e}")
            sys.exit(1)

    # Execute
    runner = ExperimentRunner()
    try:
        result = runner.run(config)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Experiment execution failed: {e}")
        sys.exit(1)

    # Persist Artifacts
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config
        atomic_write(artifact_dir / "config.json", config.model_dump_json(indent=2))
        
        # Write result
        atomic_write(result_path, result.model_dump_json(indent=2))
        logger.info(f"Artifacts successfully written to {artifact_dir}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to write artifacts: {e}")
        sys.exit(1)

    # Print Summary
    print("\n" + "="*40)
    print(f"Experiment: {result.experiment_name}")
    print(f"Dataset: {result.dataset_id} (version: {result.dataset_version})")
    
    if result.pipeline_result.poisoning_metadata:
        p_meta = result.pipeline_result.poisoning_metadata
        print(f"Poisoning: {p_meta.attack_type}")
        print(f"Poison rate: {p_meta.poison_rate}")
    else:
        print("Poisoning: Disabled")
        
    print(f"Threshold: {result.pipeline_result.threshold:.4f}")
    
    report = result.pipeline_result.evaluation_report
    print(f"Precision: {report.precision:.4f}")
    print(f"Recall: {report.recall:.4f}")
    print(f"F1: {report.f1:.4f}")
    print(f"Accuracy: {report.accuracy:.4f}")
    print(f"Experiment fingerprint: {result.experiment_fingerprint}")
    print("="*40 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
