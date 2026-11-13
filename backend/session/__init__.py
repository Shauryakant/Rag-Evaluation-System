"""Session management package for ephemeral, zero-auth document sessions."""
from backend.session.manager import DocumentSession, SessionManager, get_session_manager

__all__ = ["DocumentSession", "SessionManager", "get_session_manager"]
