"""
FastAPI Server Main Entry Point

Run this module to start the FastAPI server:
    python -m server
    
Or with uvicorn:
    uvicorn server.app:app --reload --port 8000
"""
import os
import uvicorn
from dotenv import load_dotenv

def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()

if __name__ == '__main__':
    load_env()
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    uvicorn.run(
        "server.app:app", # module path
        host=host,
        port=port,
        reload=True,
        log_level="info"  # Use string instead of int for uvicorn
    )
