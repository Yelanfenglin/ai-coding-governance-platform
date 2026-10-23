"""Support ``python -m app.ingest ...``."""

from app.ingest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
