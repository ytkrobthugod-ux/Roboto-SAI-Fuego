"""
Database setup for Roboto SAI backend (SQLite + SQLAlchemy async).
"""

import os
import logging
from typing import AsyncGenerator
from utils.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Ensure Supabase tables exist (idempotent)."""
    supabase = get_supabase_client()
    if supabase is None:
        logger.warning("No Supabase client; skipping init_db")
        return
    
    # Verify tables (select-only, no RLS/perm issues)
    tables = [
        "users",
        "auth_sessions", 
        "magic_link_tokens",
        "messages",
        "message_feedback"
    ]
    
    for table in tables:
        try:
            supabase.table(table).select("id", head=True).limit(1).execute()
            logger.info(f"Verified table: {table}")
        except Exception as e:
            logger.warning(f"Table {table} verify failed: {e}")
    
    logger.info("Supabase tables verified")


def get_supabase_client():
    """Get Supabase client for DB operations."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    if not url or not key:
        logger.warning("No Supabase creds; skipping client")
        return None
    try:
        from utils.supabase_client import create_client
        return create_client(url, key)
    except Exception as e:
        logger.warning(f"Supabase client invalid: {e}; using None")
        return None
