import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

router = APIRouter(prefix="/research", tags=["research"])

RESEARCH_DIR = Path("artifacts/research")


class ResearchArtifactItem(BaseModel):
    name: str
    relative_path: str
    category: str
    size_bytes: int
    file_type: str
    modified_at: str


@router.get("/artifacts", response_model=list[ResearchArtifactItem])
def list_research_artifacts():
    """Lists all persistent empirical research artifacts."""
    artifacts = []
    if not RESEARCH_DIR.exists():
        return artifacts

    for p in sorted(RESEARCH_DIR.rglob("*")):
        if p.is_file():
            rel = p.relative_to(RESEARCH_DIR).as_posix()
            cat = rel.split("/")[0] if "/" in rel else "root"
            stat = p.stat()
            file_type = p.suffix.lstrip(".").upper() or "FILE"
            artifacts.append(
                ResearchArtifactItem(
                    name=p.name,
                    relative_path=rel,
                    category=cat,
                    size_bytes=stat.st_size,
                    file_type=file_type,
                    modified_at=str(stat.st_mtime),
                )
            )
    return artifacts


@router.get("/data/{category}")
def get_research_category_data(category: str) -> Any:
    """Returns parsed empirical data for research views."""
    category_map = {
        "signal_diagnostics": RESEARCH_DIR / "sst2/signal_diagnostics/signal_distributions.json",
        "ablation": RESEARCH_DIR / "sst2/ablation/ablation_results.json",
        "threshold_sweep": RESEARCH_DIR / "sst2/threshold_sweep/threshold_sweep.json",
        "poison_rate_sweep": RESEARCH_DIR / "sst2/poison_rate_sweep/poison_sweep.json",
        "attack_variants": RESEARCH_DIR / "sst2/attack_variants/attack_variants_results.json",
        "target_label_sweep": RESEARCH_DIR / "sst2/target_label_sweep/target_label_results.json",
        "cross_dataset": RESEARCH_DIR / "cross_dataset/generalization_results.json",
        "baseline": RESEARCH_DIR / "sst2/baseline/baseline_result.json",
    }

    target_file = category_map.get(category)
    if not target_file or not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Research data for category '{category}' not found.")

    with open(target_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/artifacts/view/{file_path:path}")
def view_artifact_content(file_path: str):
    """Returns raw text or json content for artifact preview."""
    if ".." in file_path or file_path.startswith("/") or "\\" in file_path:
        raise HTTPException(status_code=400, detail="Invalid path traversal.")

    full_path = RESEARCH_DIR / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if file_path.endswith(".json"):
        try:
            return json.loads(content)
        except Exception:
            return PlainTextResponse(content)
    return PlainTextResponse(content)
