"""Database layer - backward compatibility wrapper.

This module provides backward compatibility by importing from database.py.
New code should import directly from polymarket_edges.database.
"""

from polymarket_edges.database import Database

__all__ = ["Database"]
