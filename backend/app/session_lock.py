import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional


class SessionBusyError(Exception):
    pass


_LOCK_TTL_SECONDS = int(os.getenv("PLATONIC_SESSION_LOCK_TTL_SECONDS", "1200"))  # 20 min
_LOCK_PREFIX = os.getenv("PLATONIC_SESSION_LOCK_PREFIX", "platonic:session_lock:")
_REDIS_URL = os.getenv("REDIS_URL")

_in_memory_locks: dict[str, asyncio.Lock] = {}
_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    if not _REDIS_URL:
        return None
    try:
        import redis.asyncio as redis  # type: ignore

        _redis = redis.from_url(_REDIS_URL, decode_responses=True)
        return _redis
    except Exception:
        # If redis client can't be imported/initialized, fall back to in-memory.
        return None


_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""


@asynccontextmanager
async def session_lock(session_id: str):
    """
    Ensures at most one in-flight request per session_id.

    - If REDIS_URL is set and redis is available, uses a distributed lock with TTL.
    - Otherwise falls back to an in-process asyncio.Lock (dev-only).
    """
    if not session_id:
        raise ValueError("Missing session_id")

    token = str(uuid.uuid4())
    redis_client = _get_redis()

    if redis_client is not None:
        key = f"{_LOCK_PREFIX}{session_id}"
        acquired = await redis_client.set(key, token, nx=True, ex=_LOCK_TTL_SECONDS)
        if not acquired:
            raise SessionBusyError("Another request is already running for this session.")
        try:
            yield
        finally:
            try:
                await redis_client.eval(_RELEASE_LUA, 1, key, token)
            except Exception:
                # Let TTL handle eventual unlock if release fails.
                pass
        return

    # In-memory fallback
    lock = _in_memory_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _in_memory_locks[session_id] = lock

    if lock.locked():
        raise SessionBusyError("Another request is already running for this session.")

    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


