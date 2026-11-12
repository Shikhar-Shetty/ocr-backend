def main() -> None:
    import uvicorn

    uvicorn.run("prestige_erp_backend.api:app", host="0.0.0.0", port=8000, reload=False)
