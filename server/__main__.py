"""
FastAPI Server Main Entry Point

Run this module to start the FastAPI server:
    python -m server
    
Or with uvicorn:
    uvicorn server.app:app --reload --port 8000
"""
import uvicorn
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()

if __name__ == '__main__':
    load_env()
    uvicorn.run(
        "server.app:app", # module path
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"  # Use string instead of int for uvicorn
    )
