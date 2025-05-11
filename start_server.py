import uvicorn

if __name__ == "__main__":
    import os
     # Cloud Run will inject PORT env var
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("server.index:app", host="0.0.0.0", port=port)
