import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_supabase_client() -> Client:
    """Get Supabase client. Use SERVICE_ROLE_KEY for server ops."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY required")
    return create_client(SUPABASE_URL, SUPABASE_KEY)