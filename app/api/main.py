import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.api.middleware import RequestIDFilter, RequestIDMiddleware
from app.api.routes import chat
from app.core.config import settings

LOG_DIR = Path(settings.base_dir) / "logs"
LOG_DIR.mkdir(exist_ok=True)

_rid_filter = RequestIDFilter()
_log_format = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] — %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)

for handler in logging.root.handlers:
    handler.addFilter(_rid_filter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("Starting up — loading pipeline components")
    yield
    logging.getLogger(__name__).info("Shutting down")


app = FastAPI(
    title="Ancient Egypt RAG API",
    description="A RAG-powered API for answering questions about Ancient Egyptian civilization.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

app.include_router(chat.router, prefix="/api")

FRONTEND_DIR = Path(settings.base_dir) / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "templates" / "index.html").read_text(encoding="utf-8")
