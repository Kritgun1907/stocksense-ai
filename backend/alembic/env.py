import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Make backend/ importable so we can import api.database ───────────────────
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# ── Import our models so Alembic knows the full schema ───────────────────────
# autogenerate compares Base.metadata against the live database
# to produce migration scripts automatically.
from api.database import Base, DATABASE_URL  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at our metadata — enables autogenerate
target_metadata = Base.metadata

# Override the URL from alembic.ini with our DATABASE_URL env var.
# This means .env or environment variables always take precedence.
# Use sync URL for Alembic (it uses its own sync connection internally).
config.set_main_option(
    "sqlalchemy.url",
    DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg2"),
)


def run_migrations_offline() -> None:
    """
    'Offline' mode — generates SQL script without connecting to DB.
    Useful for reviewing changes before applying them.
    Run with: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,       # detect column type changes
        compare_server_default=True,  # detect default value changes
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    'Online' mode — connects to DB and applies migrations.
    Uses asyncpg-compatible async engine.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = DATABASE_URL

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

