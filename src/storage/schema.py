from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector

from src.storage.models import Base


def add_missing_columns(connection: Connection) -> None:
    """Additive evolution for existing Phase 3 databases. Never drops columns."""
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            compiled = column.type.compile(dialect=connection.dialect)
            connection.execute(
                text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {compiled} NULL")
            )
        _ensure_indexes(connection, inspector, table.name)


def _ensure_indexes(connection: Connection, inspector: Inspector, table_name: str) -> None:
    existing = {index["name"] for index in inspector.get_indexes(table_name) if index.get("name")}
    table = Base.metadata.tables[table_name]
    for index in table.indexes:
        if index.name and index.name not in existing:
            index.create(bind=connection, checkfirst=True)
