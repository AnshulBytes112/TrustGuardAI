from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.datasets import router as datasets_router
from backend.api.demo import router as demo_router
from backend.api.purification import router as purification_router
from backend.api.retraining import router as retraining_router
from backend.api.samples import router as samples_router
from backend.api.scans import router as scans_router
from backend.api.stats import router as stats_router
from backend.core.database import init_db

# Initialize database tables
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="TrustGuard AI",
    description="Advanced ML Security & Data Integrity Platform for Detecting Poisoned Training Data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(demo_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(scans_router, prefix="/api")
app.include_router(samples_router, prefix="/api")
app.include_router(purification_router, prefix="/api")
app.include_router(retraining_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
