from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.demo import router as demo_router

app = FastAPI(title="TrustGuard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(demo_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}

