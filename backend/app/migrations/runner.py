"""Simple SQL migration runner for CORE.

Tracks applied migrations in a `schema_migrations` table.
Each migration is a .sql file in the migrations/sql/ directory.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "sql"


async def ensure_migrations_table(pool):
    """Create the migrations tracking table if it doesn't exist."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT NOW()
            )
        """)


async def get_applied_migrations(pool):
    """Get list of already-applied migration versions."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
        return {row["version"] for row in rows}


async def run_pending_migrations(pool):
    """Run any SQL migrations that haven't been applied yet."""
    await ensure_migrations_table(pool)
    applied = await get_applied_migrations(pool)

    if not MIGRATIONS_DIR.exists():
        logger.info("No migrations directory found at %s", MIGRATIONS_DIR)
        return

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    pending = [f for f in sql_files if f.stem not in applied]

    if not pending:
        logger.info("All %d migrations already applied", len(applied))
        return

    for migration_file in pending:
        version = migration_file.stem
        sql = migration_file.read_text()
        logger.info("Applying migration: %s", version)

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    version
                )

        logger.info("Migration %s applied successfully", version)

    logger.info("Applied %d new migrations", len(pending))
