"""Main entry point for the training dashboard backend."""

import argparse

import uvicorn

from app.api import create_app


def main() -> None:
    """Run the dashboard server."""
    parser = argparse.ArgumentParser(description="Training Dashboard Backend")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--db-path",
        default="data/dashboard.db",
        help="Path to SQLite database (default: data/dashboard.db)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Create the app
    app = create_app(db_path=args.db_path)

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()


