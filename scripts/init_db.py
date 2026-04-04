#!/usr/bin/env python
"""
Database initialization script.

Creates all tables defined in the ORM models.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --drop   # Drop & recreate (DESTROYS DATA)
"""
import asyncio
import sys

# Ensure the models are imported so Base.metadata knows about them
import app.core.models  # noqa: F401
from app.core.database import create_tables, dispose_engine, drop_tables
from app.core.config import get_settings


async def main(drop_first: bool = False):
    settings = get_settings()
    print("=" * 60)
    print("🗄️  DeepCode Agent — Database Init")
    print("=" * 60)
    print(f"  Database URL : {settings.DATABASE_URL}")
    print()

    if drop_first:
        print("⚠️  Dropping all tables...")
        await drop_tables()
        print("  ✅ Tables dropped")

    print("📦 Creating tables...")
    await create_tables()
    print("  ✅ Tables created successfully")
    print()
    print("Tables:")
    print("  • sessions")
    print("  • interaction_logs")
    print("=" * 60)

    await dispose_engine()


if __name__ == "__main__":
    drop_flag = "--drop" in sys.argv
    if drop_flag:
        confirm = input("⚠️  This will DELETE all data. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(main(drop_first=drop_flag))
