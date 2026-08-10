# TrustGuard AI

TrustGuard AI is an explainable training-data security framework for detecting and purifying potentially poisoned samples in **text training datasets**.

## Current MVP Architecture

The MVP focuses on controlled poisoning of text datasets, DistilBERT representations, and Isolation Forest / K-Means detection of poisoned samples based on multi-layer hidden state analysis inspired by FLARE.

## Technology Stack

- **ML**: Python 3.11, PyTorch, Hugging Face, scikit-learn, UMAP
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, MLflow
- **Frontend**: React, TypeScript, Vite, Tailwind CSS

## Repository Structure

```text
trustguard-ai/
├── docs/            # Project documentation and specifications
├── ml/              # ML pipelines, detection, and models
├── backend/         # FastAPI backend services
├── frontend/        # React frontend
├── configs/         # Research experiment configurations
├── scripts/         # Utility scripts
├── tests/           # Automated tests
├── artifacts/       # Local outputs and logs
├── docker/          # Dockerfiles
└── .github/         # CI workflows
```

## Local Setup

```bash
# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Set up frontend
cd frontend
npm install
cd ..

# Copy environment template
cp .env.example .env
```

## How to start Docker Compose

```bash
make docker-up
# or
docker-compose up -d
```

## How to run backend tests

```bash
make test
# or
pytest tests/backend tests/ml
```

## How to run frontend tests

```bash
make test-frontend
```

## Current Implementation Status

> This repository currently contains the project foundation. ML detection functionality will be implemented phase-by-phase according to `/docs/01_IMPLEMENTATION_PLAN.md`.
