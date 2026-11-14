"""Redis-based multi-turn conversational memory service.

Provides session-isolated, chronologically ordered chat memory with automatic TTL
expiration and history pruning.
"""

from datetime import datetime, timezone
import json
import logging
import socket
from typing import Any, Dict, List, Optional
import redis

from app.core.config import Settings, get_settings
from app.utils.exceptions import MemoryException

logger = logging.getLogger("palmind.memory")


class MemoryService:
    """Manages persistent session-based conversation memory using Redis."""

    def __init__(
        self,
        client: Optional[redis.Redis] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._history_limit = self.settings.REDIS_HISTORY_LIMIT
        self._ttl_seconds = self.settings.REDIS_SESSION_TTL_SECONDS
        self._client = client or self._init_client()

    def _init_client(self) -> redis.Redis:
        """Initialize Redis client with fallback to Fakeredis when remote host is unreachable."""
        host = self.settings.REDIS_HOST
        port = self.settings.REDIS_PORT
        is_port_open = False

        # Fast socket probe to avoid Windows TCP SYN retransmit delays
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.3)
            res = sock.connect_ex((host, port))
            is_port_open = (res == 0)
            sock.close()
        except Exception:
            is_port_open = False

        if is_port_open:
            try:
                client = redis.Redis(
                    host=host,
                    port=port,
                    db=self.settings.REDIS_DB,
                    socket_timeout=2,
                    socket_connect_timeout=2,
                    decode_responses=True,
                )
                client.ping()
                logger.info("Connected to Redis at %s:%d (db=%d)", host, port, self.settings.REDIS_DB)
                return client
            except (redis.ConnectionError, redis.TimeoutError) as exc:
                logger.warning("Redis ping failed (%s). Falling back to Fakeredis.", exc)

        logger.info("Remote Redis unavailable at %s:%d. Using Fakeredis session memory engine.", host, port)
        try:
            import fakeredis
            return fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            raise MemoryException(
                message=f"Could not connect to Redis at {host}:{port} and fakeredis is not installed.",
                details={"host": host, "port": port},
            )

    def _session_key(self, session_id: str) -> str:
        """Return namespaced Redis key for session history."""
        clean_id = session_id.strip()
        return f"palmind:session:{clean_id}:history"

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message turn to the session history in Redis.

        Args:
            session_id: Unique conversation identifier.
            role: 'user' or 'assistant'.
            content: Text message content.
        """
        if not session_id or not content or not content.strip():
            return

        key = self._session_key(session_id)
        content_clean = content.strip()

        try:
            # Duplicate prevention: Check if the most recent message has identical role and content
            last_item_json = self._client.lindex(key, -1)
            if last_item_json:
                try:
                    last_item = json.loads(last_item_json)
                    if last_item.get("role") == role and last_item.get("content") == content_clean:
                        logger.debug("Skipping duplicate message for session %s (role=%s)", session_id, role)
                        return
                except json.JSONDecodeError:
                    pass

            # Create message payload
            message_payload = {
                "role": role,
                "content": content_clean,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            serialized = json.dumps(message_payload)

            # RPUSH to append chronologically
            self._client.rpush(key, serialized)

            # Enforce history limit (keep last 2 * limit entries: pairs of user+assistant)
            max_entries = max(2, self._history_limit * 2)
            self._client.ltrim(key, -max_entries, -1)

            # Refresh session expiration TTL
            self._client.expire(key, self._ttl_seconds)
            logger.debug("Stored message for session %s (role=%s, ttl=%ds)", session_id, role, self._ttl_seconds)

        except Exception as exc:
            logger.error("Failed to add message to Redis session %s: %s", session_id, exc)
            raise MemoryException(
                message=f"Failed to record conversation message in Redis: {str(exc)}",
                details={"session_id": session_id, "error": str(exc)},
            ) from exc

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Convenience method to store a full user-assistant interaction turn."""
        self.add_message(session_id=session_id, role="user", content=user_message)
        self.add_message(session_id=session_id, role="assistant", content=assistant_message)

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve chronological conversation history for a session.

        Args:
            session_id: Unique conversation identifier.
            limit: Optional override for number of past turns to retrieve.

        Returns:
            List of message dicts: [{'role': 'user', 'content': '...', 'timestamp': '...'}, ...]
        """
        if not session_id or not session_id.strip():
            return []

        key = self._session_key(session_id)

        try:
            raw_entries = self._client.lrange(key, 0, -1)
            if not raw_entries:
                return []

            history: List[Dict[str, str]] = []
            for entry in raw_entries:
                try:
                    parsed = json.loads(entry)
                    history.append(parsed)
                except json.JSONDecodeError:
                    continue

            # Refresh TTL on access
            self._client.expire(key, self._ttl_seconds)

            # Apply turn limit if specified
            max_turns = limit or self._history_limit
            max_messages = max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]

            return history

        except Exception as exc:
            logger.error("Failed to read Redis session history for %s: %s", session_id, exc)
            raise MemoryException(
                message=f"Failed to retrieve conversation history from Redis: {str(exc)}",
                details={"session_id": session_id, "error": str(exc)},
            ) from exc

    def clear_session(self, session_id: str) -> bool:
        """Purge conversation history for a session."""
        key = self._session_key(session_id)
        try:
            deleted = self._client.delete(key)
            return bool(deleted)
        except Exception as exc:
            logger.error("Failed to clear Redis session %s: %s", session_id, exc)
            raise MemoryException(
                message=f"Failed to clear Redis session history: {str(exc)}",
                details={"session_id": session_id, "error": str(exc)},
            ) from exc

    def get_session_ttl(self, session_id: str) -> int:
        """Get remaining TTL in seconds for a session key (-2 if expired/nonexistent, -1 if no TTL)."""
        key = self._session_key(session_id)
        try:
            return int(self._client.ttl(key))
        except Exception as exc:
            logger.error("Failed to check TTL for session %s: %s", session_id, exc)
            return -2

    def _booking_draft_key(self, session_id: str) -> str:
        """Return namespaced Redis key for session booking draft."""
        clean_id = session_id.strip()
        return f"palmind:session:{clean_id}:booking_draft"

    def get_booking_draft(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve in-progress booking draft from Redis."""
        if not session_id or not session_id.strip():
            return None
        key = self._booking_draft_key(session_id)
        try:
            val = self._client.get(key)
            if not val:
                return None
            return json.loads(val)
        except Exception as exc:
            logger.error("Failed to read booking draft for %s: %s", session_id, exc)
            return None

    def save_booking_draft(self, session_id: str, draft: Dict[str, Any]) -> None:
        """Persist incomplete booking draft in Redis with session TTL."""
        if not session_id or not session_id.strip():
            return
        key = self._booking_draft_key(session_id)
        try:
            serialized = json.dumps(draft)
            self._client.set(key, serialized, ex=self._ttl_seconds)
        except Exception as exc:
            logger.error("Failed to save booking draft for %s: %s", session_id, exc)
            raise MemoryException(
                message=f"Failed to persist booking draft in Redis: {str(exc)}",
                details={"session_id": session_id, "error": str(exc)},
            ) from exc

    def clear_booking_draft(self, session_id: str) -> bool:
        """Purge incomplete booking draft from Redis."""
        if not session_id or not session_id.strip():
            return False
        key = self._booking_draft_key(session_id)
        try:
            return bool(self._client.delete(key))
        except Exception as exc:
            logger.error("Failed to delete booking draft for %s: %s", session_id, exc)
            return False


# Singleton instance for application reuse
memory_service = MemoryService()


def get_memory_service() -> MemoryService:
    """FastAPI dependency provider for MemoryService."""
    return memory_service
