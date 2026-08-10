from fastapi import FastAPI

app = FastAPI(title="TrustGuard AI")

@app.get("/health")
def health():
    return {"status": "ok"}
