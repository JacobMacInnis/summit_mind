import uvicorn

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Cloud Run will inject PORT env var
    uvicorn.run("server.index:app", host="0.0.0.0", port=port)
