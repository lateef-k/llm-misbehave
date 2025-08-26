"""
SQLite-based key-value cache using SQLAlchemy async.

This module provides a simple async key-value cache backed by SQLite
with JSON serialization support.
"""

import pickle
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, PickleType, String, select
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class CacheEntry(Base):
    """Key-value cache entry model."""

    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[object] = mapped_column(PickleType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# Global async engine and session maker
async_engine = create_async_engine(
    "sqlite+aiosqlite:///cache.db",
    echo=False,
)

SessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Initialize the database by creating all tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class AsyncCache:
    """Async key-value cache using SQLite."""

    async def get(self, key: str) -> Optional[Any]:
        async with SessionLocal() as session:
            stmt = select(CacheEntry).where(CacheEntry.key == key)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry is None:
                print("Cache miss")
                return None
            print("Cache hit")
            return entry.value

    async def set(self, key: str, value: Any) -> None:
        async with SessionLocal() as session:
            # Check if entry exists
            stmt = select(CacheEntry).where(CacheEntry.key == key)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()

            if entry is None:
                # Create new entry
                entry = CacheEntry(key=key, value=value)
                session.add(entry)
            else:
                # Update existing entry
                entry.value = value

            await session.commit()

    async def delete(self, key: str) -> bool:
        async with SessionLocal() as session:
            stmt = select(CacheEntry).where(CacheEntry.key == key)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()

            if entry is None:
                return False

            await session.delete(entry)
            await session.commit()
            return True

    async def clear(self) -> int:
        async with SessionLocal() as session:
            stmt = select(CacheEntry)
            result = await session.execute(stmt)
            entries = result.scalars().all()
            count = len(entries)

            for entry in entries:
                await session.delete(entry)

            await session.commit()
            return count


if __name__ == "__main__":
    import asyncio

    async def main():
        """Create the database tables."""
        await init_db()
        print("Database tables created successfully!")

    asyncio.run(main())
