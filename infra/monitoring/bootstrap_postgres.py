"""Initialize a monitor login using psql's native, local password prompt."""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from psycopg import sql
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps/rag-api"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin", default="postgres", help="Existing PostgreSQL administrator")
    parser.add_argument("--psql", default="psql", help="PostgreSQL psql executable")
    args = parser.parse_args()
    executable = shutil.which(args.psql)
    if executable is None:
        parser.error("psql was not found; pass --psql with its executable path")
    from app.settings import Settings

    settings = Settings(_env_file=ROOT / "apps/rag-api/.env")
    url = make_url(settings.database_url.get_secret_value())
    directory = ROOT / "output/monitoring_secrets"
    directory.mkdir(parents=True, exist_ok=True)
    password_file = directory / "postgres_password"
    metadata_file = directory / "postgres.json"
    role = "embedded_monitor"
    print(f"Target: {url.host}:{url.port or 5432}/{url.database}; administrator: {args.admin}")
    if password_file.exists() and metadata_file.exists():
        print("Existing monitoring credentials retained; no role changes requested.")
        return
    password = secrets.token_urlsafe(32)
    statements = sql.SQL("""
BEGIN;
CREATE ROLE {role} LOGIN PASSWORD {password} NOSUPERUSER NOCREATEDB NOCREATEROLE;
GRANT pg_monitor TO {role};
GRANT CONNECT ON DATABASE {database} TO {role};
ALTER ROLE {role} SET default_transaction_read_only = on;
COMMIT;
""").format(role=sql.Identifier(role), password=sql.Literal(password),
            database=sql.Identifier(url.database))
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".sql",
                                     dir=directory, delete=False) as script:
        script.write(statements.as_string())
        script_path = Path(script.name)
    try:
        completed = subprocess.run([
            executable, "--no-psqlrc", "--quiet", "--set=ON_ERROR_STOP=1", "--password",
            "--host", str(url.host), "--port", str(url.port or 5432),
            "--username", args.admin, "--dbname", str(url.database), "--file", str(script_path),
        ], check=False)
    finally:
        script_path.unlink(missing_ok=True)
    if completed.returncode:
        raise SystemExit("PostgreSQL setup failed; no monitoring credentials were saved.")
    password_file.write_text(password, encoding="utf-8")
    metadata_file.write_text(json.dumps({
        "user": role, "uri": f"{url.host}:{url.port or 5432}/{url.database}?sslmode=disable",
        "password_file": str(password_file),
    }, indent=2), encoding="utf-8")
    print("Read-only monitor configured. Credentials are in ignored output/monitoring_secrets.")
    print("No business tables or pg_hba.conf were changed.")


if __name__ == "__main__":
    main()
