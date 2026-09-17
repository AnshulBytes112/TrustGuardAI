from __future__ import annotations

import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.schemas.live import (
    LiveInvestigationRequest,
    LiveJobResponse,
    LiveJobSummary,
    LiveSSEEvent,
)
from backend.services.live_investigation_service import LiveInvestigationService

router = APIRouter(prefix="/live", tags=["live-investigation"])


@router.post(
    "/investigate",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start Real-Time Research Investigation",
)
def start_live_investigation(request: LiveInvestigationRequest):
    """
    Kicks off an asynchronous, non-blocking research investigation pipeline.
    Emits real-time SSE events on /api/live/stream/{job_id}.
    """
    job = LiveInvestigationService.start_investigation(request)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "dataset_id": request.dataset_id,
        "attack_type": request.attack_type,
    }


@router.get(
    "/stream/{job_id}",
    summary="Connect Live SSE Event Stream",
    response_class=StreamingResponse,
)
async def stream_live_events(job_id: str):
    """
    Subscribes to Server-Sent Events (SSE) for the specified live investigation job.
    Replaying previous events ensures reliable reconnection across page refreshes.
    """
    job = LiveInvestigationService.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live investigation job '{job_id}' not found.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for ev in LiveInvestigationService.subscribe_events(job_id):
                payload = json.dumps(ev.model_dump(mode="json"))
                yield f"id: {ev.event_id}\nevent: {ev.event_type}\ndata: {payload}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)})
            yield f"event: ERROR\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/jobs/{job_id}",
    response_model=LiveJobResponse,
    summary="Get Job State & Inspection Results",
)
def get_live_job(job_id: str):
    """
    Returns the authoritative backend state of a live investigation job,
    including granular sample inspections, retraining reports, and baseline comparisons.
    """
    job = LiveInvestigationService.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live investigation job '{job_id}' not found.",
        )
    return job


@router.get(
    "/jobs",
    response_model=list[LiveJobSummary],
    summary="List Recent Live Investigation Jobs",
)
def list_live_jobs():
    """
    Lists all recent live investigation jobs from the active backend session.
    """
    return LiveInvestigationService.list_jobs()
