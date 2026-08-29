import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import hashlib

from ml.experiments.runner import ExperimentRunner
from ml.experiments.schemas import ExperimentConfig, ExperimentResult

router = APIRouter(prefix="/demo", tags=["demo"])

CONFIG_DIR = Path("configs/experiments")
ARTIFACT_DIR = Path("artifacts/experiments")
DATASET_DIR = Path("artifacts/datasets")


class ExperimentListItem(BaseModel):
    experiment_name: str
    experiment_fingerprint: str
    dataset_id: str
    dataset_version: str
    poisoning_enabled: bool
    status: str


class RunExperimentRequest(BaseModel):
    config_name: str
    custom_dataset_id: str | None = None

class DatasetUploadResponse(BaseModel):
    dataset_id: str
    dataset_version: str
    filename: str
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    label_mode: str


@router.get("/experiments", response_model=list[ExperimentListItem])
def list_experiments():
    """Returns a list of completed experiments from the artifacts directory."""
    results = []
    if not ARTIFACT_DIR.exists():
        return results

    for exp_dir in ARTIFACT_DIR.iterdir():
        if exp_dir.is_dir():
            result_file = exp_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Validate minimally to avoid full load if possible, 
                    # but using ExperimentResult ensures correctness.
                    # For performance, we just pull the needed fields from JSON
                    is_poisoned = False
                    if "pipeline_result" in data and "poisoning_metadata" in data["pipeline_result"]:
                        is_poisoned = data["pipeline_result"]["poisoning_metadata"] is not None

                    results.append(
                        ExperimentListItem(
                            experiment_name=data.get("experiment_name", "unknown"),
                            experiment_fingerprint=data.get("experiment_fingerprint", exp_dir.name),
                            dataset_id=data.get("dataset_id", "unknown"),
                            dataset_version=data.get("dataset_version", "unknown"),
                            poisoning_enabled=is_poisoned,
                            status="completed"
                        )
                    )
                except Exception:  # noqa: BLE001, S110
                    pass

    return results


@router.get("/experiments/{fingerprint}", response_model=ExperimentResult)
def get_experiment(fingerprint: str):
    """Returns the full ExperimentResult for a given fingerprint."""
    # Prevent directory traversal
    if "/" in fingerprint or "\\" in fingerprint or ".." in fingerprint:
        raise HTTPException(status_code=400, detail="Invalid fingerprint")

    result_file = ARTIFACT_DIR / fingerprint / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Experiment artifact not found")

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            result_json = f.read()
        return ExperimentResult.model_validate_json(result_json)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load experiment result: {e}")


def atomic_write(filepath: Path, content: str) -> None:
    """Writes content to filepath atomically using a .tmp file."""
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_path.replace(filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


@router.post("/run")
def run_experiment(req: RunExperimentRequest):
    """
    Runs the selected baseline experiment.
    If an artifact already exists, returns it directly.
    """
    # Prevent directory traversal
    if "/" in req.config_name or "\\" in req.config_name or ".." in req.config_name:
        raise HTTPException(status_code=400, detail="Invalid config name")
        
    config_file = CONFIG_DIR / f"{req.config_name}.json"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Configuration {req.config_name} not found")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_json = f.read()
            config = ExperimentConfig.model_validate_json(config_json)
            
        if req.custom_dataset_id:
            # Prevent directory traversal
            if "/" in req.custom_dataset_id or "\\" in req.custom_dataset_id or ".." in req.custom_dataset_id:
                raise HTTPException(status_code=400, detail="Invalid custom dataset id")
                
            custom_dataset_path = DATASET_DIR / req.custom_dataset_id / "dataset.jsonl"
            if not custom_dataset_path.exists():
                raise HTTPException(status_code=404, detail="Custom dataset not found")
                
            # Programmatically override the dataset configuration
            config_dict = config.model_dump()
            config_dict["dataset"]["path"] = str(custom_dataset_path)
            config_dict["dataset"]["configuration"]["dataset_id"] = req.custom_dataset_id
            config_dict["dataset"]["configuration"]["dataset_version"] = "v1"
            config = ExperimentConfig.model_validate(config_dict)
            
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to parse or validate configuration: {e}")

    fingerprint = config.compute_fingerprint()
    artifact_dir = ARTIFACT_DIR / fingerprint
    result_path = artifact_dir / "result.json"

    # If already computed, return the existing artifact
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                existing_result = f.read()
            return {"status": "loaded", "fingerprint": fingerprint, "result": json.loads(existing_result)}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to read existing artifact: {e}")

    # Otherwise, run it
    runner = ExperimentRunner()
    try:
        result = runner.run(config)
    except Exception as e:  # noqa: BLE001
        # e.g., "Validation split must contain both clean and poisoned samples"
        return {"status": "failed", "error": str(e), "fingerprint": "", "result": None}

    # Save artifacts
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(artifact_dir / "config.json", config.model_dump_json(indent=2))
        atomic_write(result_path, result.model_dump_json(indent=2))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to write artifacts: {e}")

    return {"status": "completed", "fingerprint": fingerprint, "result": json.loads(result.model_dump_json())}

@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a custom JSONL dataset, validates it, and stores it deterministically."""
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")
        
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
        
    # Calculate deterministic identity
    file_hash = hashlib.sha256(content).hexdigest()
    dataset_id = f"custom_{file_hash[:12]}"
    
    # Validate content line by line
    lines = content.decode("utf-8").strip().split("\n")
    
    total_samples = 0
    train_samples = 0
    val_samples = 0
    test_samples = 0
    labels = set()
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Invalid JSON at line {i+1}")
            
        if "text" not in data or "split" not in data:
            raise HTTPException(status_code=400, detail=f"Missing required fields ('text', 'split') at line {i+1}")
            
        split = data["split"].upper()
        if split == "TRAIN":
            train_samples += 1
        elif split == "VALIDATION":
            val_samples += 1
        elif split == "TEST":
            test_samples += 1
        else:
            raise HTTPException(status_code=400, detail=f"Invalid split '{split}' at line {i+1}")
            
        if "label" in data and data["label"] is not None:
            labels.add(data["label"])
            
        total_samples += 1
        
    if train_samples == 0 or val_samples == 0 or test_samples == 0:
        raise HTTPException(status_code=400, detail="Dataset must contain TRAIN, VALIDATION, and TEST splits")
        
    label_mode = "UNLABELLED"
    if len(labels) > 0:
        label_mode = "FULLY_LABELLED" # simplified for demo
        
    # Save the dataset
    dataset_dir = DATASET_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.jsonl"
    
    if not dataset_path.exists():
        with open(dataset_path, "wb") as f:
            f.write(content)
            
    return DatasetUploadResponse(
        dataset_id=dataset_id,
        dataset_version="v1",
        filename=file.filename,
        total_samples=total_samples,
        train_samples=train_samples,
        validation_samples=val_samples,
        test_samples=test_samples,
        label_mode=label_mode
    )
