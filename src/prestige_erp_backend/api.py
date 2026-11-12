import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Prestige ERP Catalogue OCR", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
EXTRACTOR = Path(__file__).parent / "extractor" / "catalogue_extractor.py"
MAX_FILE_SIZE = 50 * 1024 * 1024
EXTRACTION_TIMEOUT_SECONDS = 180
TEMP_ROOT = Path(tempfile.gettempdir()) / "prestige-erp-catalogue-ocr"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
TARGET_BYTES = 4 * 1024 * 1024
ILOVEPDF_PUBLIC_KEY_ENV = "ILOVEAPI_PUBLIC_KEY"
ILOVEPDF_SECRET_KEY_ENV = "ILOVEAPI_SECRET_KEY"


def _job_dir(token: str) -> Path:
    if not token or Path(token).name != token:
        raise HTTPException(status_code=404, detail="OCR job not found")
    directory = TEMP_ROOT / token
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="OCR job not found or expired")
    return directory


def _compress_pdf(source: Path, destination: Path) -> None:
    public_key = os.getenv(ILOVEPDF_PUBLIC_KEY_ENV)
    secret_key = os.getenv(ILOVEPDF_SECRET_KEY_ENV)
    if not public_key or not secret_key:
        raise HTTPException(
            status_code=503,
            detail=f"{ILOVEPDF_PUBLIC_KEY_ENV} and {ILOVEPDF_SECRET_KEY_ENV} must be configured",
        )
    try:
        from ilovepdf import CompressTask

        task = CompressTask(public_key=public_key, secret_key=secret_key)
        task.compression_level = "extreme"
        task.add_file(str(source))
        task.execute()
        task.download(str(destination.parent))
        downloaded = destination.parent / source.name
        if downloaded != destination:
            downloaded.replace(destination)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"iLoveAPI PDF compression failed: {error}") from error
    if not destination.exists():
        raise HTTPException(status_code=502, detail="iLoveAPI did not return a compressed PDF")


def _compressed_pdf(directory: Path) -> Path:
    compressed = directory / "compressed.pdf"
    if not compressed.exists():
        raise HTTPException(status_code=404, detail="Compressed PDF is unavailable")
    return compressed


def _run_extractor(pdf_path: Path, output_dir: Path, job_id: str, start_page: int, end_page: int, pages: str) -> dict:
    command = [
        sys.executable, str(EXTRACTOR), "--pdf", str(pdf_path),
        "--output-dir", str(output_dir), "--job-id", job_id,
        "--start-page", str(start_page), "--end-page", str(end_page),
    ]
    if pages:
        command.extend(["--pages", pages])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=EXTRACTION_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        raise HTTPException(status_code=504, detail="Catalogue extraction timed out") from error
    result_path = output_dir / "extraction_result.json"
    if completed.returncode != 0 or not result_path.exists():
        detail = completed.stderr.strip() or completed.stdout.strip() or "Catalogue extraction failed"
        raise HTTPException(status_code=422, detail=detail[-4000:])
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=502, detail="Extractor returned invalid JSON") from error


def _inline_assets(output_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "filename": path.name,
            "contentType": "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
            "content": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        for path in output_dir.iterdir()
        if path.is_file() and not path.name.endswith(".json")
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/catalogue/prepare")
async def prepare_catalogue(
    file: Annotated[UploadFile, File(description="PDF catalogue")],
) -> dict:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF catalogues are supported")
    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="PDF exceeds the 50 MB limit")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid PDF document")

    token = uuid4().hex
    directory = TEMP_ROOT / token
    directory.mkdir()
    original = directory / "original.pdf"
    compressed = directory / "compressed.pdf"
    original.write_bytes(contents)
    try:
        await asyncio.to_thread(_compress_pdf, original, compressed)
        original.unlink(missing_ok=True)
        import fitz
        page_count = len(fitz.open(compressed))
        compressed_bytes = compressed.stat().st_size
        return {
            "token": token,
            "originalBytes": len(contents),
            "compressedBytes": compressed_bytes,
            "targetBytes": TARGET_BYTES,
            "targetReached": compressed_bytes <= TARGET_BYTES,
            "pageCount": page_count,
        }
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


@app.post("/api/catalogue/extract/{token}")
async def extract_catalogue(
    token: str,
    job_id: Annotated[str, Form()],
    start_page: Annotated[int, Form()] = 1,
    end_page: Annotated[int, Form()] = 0,
    pages: Annotated[str, Form()] = "",
) -> dict:
    directory = _job_dir(token)
    if start_page < 1 or end_page < 0:
        raise HTTPException(status_code=422, detail="Invalid page range")
    output_dir = directory / "images"
    output_dir.mkdir(exist_ok=True)
    result = await asyncio.to_thread(
        _run_extractor, _compressed_pdf(directory), output_dir, job_id, start_page, end_page, pages
    )
    return {"result": result}


@app.post("/api/catalogue/extract")
async def extract_catalogue_legacy(
    file: Annotated[UploadFile, File(description="PDF catalogue")],
    job_id: Annotated[str, Form()],
    start_page: Annotated[int, Form()] = 1,
    end_page: Annotated[int, Form()] = 0,
    pages: Annotated[str, Form()] = "",
) -> dict:
    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE or not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=422, detail="Invalid PDF upload")
    directory = TEMP_ROOT / uuid4().hex
    directory.mkdir()
    original = directory / "original.pdf"
    compressed = directory / "compressed.pdf"
    output_dir = directory / "images"
    output_dir.mkdir()
    original.write_bytes(contents)
    try:
        await asyncio.to_thread(_compress_pdf, original, compressed)
        result = await asyncio.to_thread(_run_extractor, compressed, output_dir, job_id, start_page, end_page, pages)
        return {"result": result, "assets": _inline_assets(output_dir)}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@app.get("/api/catalogue/{token}/source.pdf")
async def download_compressed_pdf(token: str) -> FileResponse:
    return FileResponse(_compressed_pdf(_job_dir(token)), media_type="application/pdf", filename="source.pdf")


@app.get("/api/catalogue/{token}/assets/{filename}")
async def download_asset(token: str, filename: str) -> FileResponse:
    if Path(filename).name != filename or filename.endswith(".json"):
        raise HTTPException(status_code=404, detail="Asset not found")
    path = _job_dir(token) / "images" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.delete("/api/catalogue/{token}")
async def delete_catalogue_temp(token: str) -> dict[str, bool]:
    directory = _job_dir(token)
    shutil.rmtree(directory, ignore_errors=True)
    return {"deleted": True}
