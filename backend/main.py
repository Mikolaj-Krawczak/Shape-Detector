"""
Punkt wejścia API: FastAPI + integracja z silnikiem Stockfish (MVP).
"""
from fastapi import FastAPI

app = FastAPI(title="ML-Chess API")


@app.get("/health")
def health():
    """Prosty endpoint do sprawdzenia, czy serwer działa."""
    return {"status": "ok"}
