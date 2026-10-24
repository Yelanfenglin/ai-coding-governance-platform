-- Run once with psql as a database administrator; no application tables change.
-- The final psql command prompts for a password without saving it in this file.
CREATE ROLE embedded_monitor LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
GRANT pg_monitor TO embedded_monitor;
GRANT CONNECT ON DATABASE rag_kb TO embedded_monitor;
ALTER ROLE embedded_monitor SET default_transaction_read_only = on;
\password embedded_monitor
