from fastapi import FastAPI
from app.config import APP_NAME, APP_VERSION
from app.api.routes import router

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Modular Retrieval-Augmented Generation API"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to RAG Assistant API",
        "docs": "/docs"
    }