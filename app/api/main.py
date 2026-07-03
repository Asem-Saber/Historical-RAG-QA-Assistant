from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — pipeline is lazily initialized on first request."""
    yield


app = FastAPI(
    title="Ancient Egypt RAG API",
    description="A RAG-powered API for answering questions about Ancient Egyptian civilization.",
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

app.include_router(chat.router, prefix="/api")
