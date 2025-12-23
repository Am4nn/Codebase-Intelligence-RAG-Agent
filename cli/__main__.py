"""
CLI Main Entry Point

Run this module to start the interactive CLI:
    python -m cli
"""
import asyncio
import logging
from dotenv import load_dotenv
from cli.main import main


def configure_logging(level: int = logging.INFO) -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


if __name__ == '__main__':
    configure_logging(logging.INFO)
    load_env()
    asyncio.run(main())
