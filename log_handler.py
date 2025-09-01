#!/usr/bin/env python3
"""
SQLite logging handler for agent applications.

This module provides a custom logging handler that writes log records
to a SQLite database with structured fields for analysis and querying.
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional
import re
from typing import Any


def sanitize_error_for_logging(error_msg: Any) -> str:
    """Remove API keys from error messages"""
    # Remove common API key patterns
    patterns = [
        # OpenAI keys (various formats)
        r'sk-[a-zA-Z0-9]{20,}',
        r'sk-proj-[a-zA-Z0-9]{25,}',

        # Anthropic keys
        r'sk-ant-[a-zA-Z0-9-]{40,}',

        # Google keys (multiple formats)
        r'AIza[a-zA-Z0-9_-]{35}',
        r'gsk_[a-zA-Z0-9]{40,}',
        r'ya29\.[a-zA-Z0-9_-]+',

        # AWS keys
        r'AKIA[0-9A-Z]{16}',
        r'aws_access_key_id=[A-Z0-9]+',

        # Generic patterns (case-insensitive)
        r'(?i)api[_-]?key\s*[=:]\s*["\']?[a-zA-Z0-9_-]{8,}["\']?',
        r'(?i)authorization\s*:\s*[^\s,]+',
        r'(?i)bearer\s+[a-zA-Z0-9_.-]+',
        r'(?i)token\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',
        r'(?i)secret\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',
        r'(?i)password\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{8,}["\']?',

        # JSON context
        r'"(?:api_key|token|secret|password)"\s*:\s*"[^"]+',

        # URL parameters
        r'[?&](?:api_key|token|secret|password)=[^&\s]+',

        # GitHub tokens
        r'ghp_[a-zA-Z0-9]{36}',
        r'gho_[a-zA-Z0-9]{36}',

        # Slack tokens
        r'xox[bpoa]-[0-9]+-[0-9]+-[a-zA-Z0-9]+',
    ]

    sanitized = str(error_msg)
    for pattern in patterns:
        sanitized = re.sub(pattern, '[API_KEY_REDACTED]', sanitized, flags=re.IGNORECASE)

    return sanitized


class SQLiteLogHandler(logging.Handler):
    """Custom logging handler that writes logs to SQLite database."""

    def __init__(self, db_path: str = "agent_logs.db"):
        """
        Initialize the SQLite logging handler.

        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__()
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with logs table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function_name TEXT,
                line_number INTEGER,
                step_name TEXT,
                agent_session TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on commonly queried fields
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_session_timestamp
            ON agent_logs(agent_session, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_step_name
            ON agent_logs(step_name)
        """)

        conn.commit()
        conn.close()

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to the SQLite database.

        Args:
            record: The log record to emit
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract additional fields from record if available
            step_name = getattr(record, 'step_name', None)
            agent_session = getattr(record, 'agent_session', None)

            # Sanitize the log message to remove any API keys or sensitive data
            sanitized_message = sanitize_error_for_logging(record.getMessage())

            cursor.execute("""
                INSERT INTO agent_logs
                (timestamp, level, logger_name, message, module, function_name,
                 line_number, step_name, agent_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.fromtimestamp(record.created).isoformat(),
                record.levelname,
                record.name,
                sanitized_message,  # Use sanitized message
                record.module,
                record.funcName,
                record.lineno,
                step_name,
                agent_session
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            # Use handleError to avoid recursive logging issues
            self.handleError(record)

    def query_logs(self,
                   session_id: Optional[str] = None,
                   step_name: Optional[str] = None,
                   level: Optional[str] = None,
                   limit: int = 100) -> list:
        """
        Query logs from the database with optional filters.

        Args:
            session_id: Filter by agent session ID
            step_name: Filter by step name
            level: Filter by log level (INFO, WARNING, ERROR, etc.)
            limit: Maximum number of records to return

        Returns:
            List of log records as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build dynamic query
            query = """
                SELECT timestamp, level, logger_name, message, module,
                       function_name, line_number, step_name, agent_session
                FROM agent_logs
                WHERE 1=1
            """
            params = []

            if session_id:
                query += " AND agent_session = ?"
                params.append(session_id)

            if step_name:
                query += " AND step_name = ?"
                params.append(step_name)

            if level:
                query += " AND level = ?"
                params.append(level.upper())

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            columns = ['timestamp', 'level', 'logger_name', 'message', 'module',
                      'function_name', 'line_number', 'step_name', 'agent_session']

            logs = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return logs

        except Exception as e:
            # Return empty list on error to avoid breaking the application
            print(f"Error querying logs: {e}")
            return []

    def get_session_stats(self, session_id: str) -> dict:
        """
        Get statistics for a specific session.

        Args:
            session_id: The agent session ID

        Returns:
            Dictionary with session statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get total count by level
            cursor.execute("""
                SELECT level, COUNT(*)
                FROM agent_logs
                WHERE agent_session = ?
                GROUP BY level
            """, (session_id,))

            level_counts = dict(cursor.fetchall())

            # Get step counts
            cursor.execute("""
                SELECT step_name, COUNT(*)
                FROM agent_logs
                WHERE agent_session = ? AND step_name IS NOT NULL
                GROUP BY step_name
            """, (session_id,))

            step_counts = dict(cursor.fetchall())

            # Get time range
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM agent_logs
                WHERE agent_session = ?
            """, (session_id,))

            min_time, max_time, total_count = cursor.fetchone()

            conn.close()

            return {
                'session_id': session_id,
                'total_logs': total_count,
                'level_counts': level_counts,
                'step_counts': step_counts,
                'time_range': {
                    'start': min_time,
                    'end': max_time
                }
            }

        except Exception as e:
            print(f"Error getting session stats: {e}")
            return {}


def setup_sqlite_logging(logger_name: str,
                        db_path: str = "agent_logs.db",
                        level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with SQLite handler.

    Args:
        logger_name: Name for the logger
        db_path: Path to SQLite database file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Add SQLite handler
    sqlite_handler = SQLiteLogHandler(db_path)
    sqlite_handler.setLevel(level)
    sqlite_formatter = logging.Formatter('%(message)s')
    sqlite_handler.setFormatter(sqlite_formatter)

    logger.addHandler(sqlite_handler)
    logger.propagate = False

    return logger


if __name__ == "__main__":
    # Demo usage
    print("🧪 Testing SQLiteLogHandler...")

    # Create test logger
    logger = setup_sqlite_logging("test_logger", "test_logs.db")

    # Log some test messages
    logger.info("Test info message", extra={
        'step_name': 'test_step',
        'agent_session': 'demo_session'
    })

    logger.warning("Test warning message", extra={
        'step_name': 'test_step',
        'agent_session': 'demo_session'
    })

    logger.error("Test error message", extra={
        'step_name': 'error_step',
        'agent_session': 'demo_session'
    })

    # Query the logs
    handler = SQLiteLogHandler("test_logs.db")
    logs = handler.query_logs(session_id="demo_session", limit=10)

    print(f"\n📋 Retrieved {len(logs)} log entries:")
    for log in logs:
        print(f"  {log['timestamp'][:19]} | {log['level']} | {log['step_name']} | {log['message']}")

    # Get session stats
    stats = handler.get_session_stats("demo_session")
    print(f"\n📊 Session stats: {stats}")

    print("✅ SQLiteLogHandler test completed!")