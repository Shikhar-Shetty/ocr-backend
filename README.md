## Prestige ERP backend

This service exposes the catalogue OCR pipeline used by the Next.js application.
It does not persist catalogue records. Uploaded PDFs and generated assets are
temporary files on the service disk.

### Run locally

```bash
uv sync
uv run uvicorn prestige_erp_backend.api:app --host 127.0.0.1 --port 8000
```

The browser uploads directly to
`POST /api/catalogue/prepare`, which compresses the PDF through iLoveAPI and
returns a temporary token and compression statistics. Next.js then downloads
the compressed source server-to-server and requests selected-page extraction
through `POST /api/catalogue/extract/{token}`. Generated assets are downloaded
individually and stored in Next.js's existing catalogue job directory. The
temporary backend directory is deleted after extraction.

The browser must be configured with the public Render URL using
`NEXT_PUBLIC_CATALOGUE_OCR_API_URL`. The server-side Next.js environment uses
`CATALOGUE_OCR_API_URL` for downloading and extracting.

The compression step uses the iLoveAPI Python SDK with its `extreme`
compression level. Configure `ILOVEAPI_PUBLIC_KEY` and `ILOVEAPI_SECRET_KEY`
in the Render environment. Deploy the service as a standard Python web service;
no native PDF compressor or Docker image is required.

The extractor uses embedded PDF text and images without extra OCR packages.
Install PaddleOCR and its supported runtime separately when scanned-document OCR
is required; the extractor detects it automatically when available.