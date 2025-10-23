import sqlite3
from typing import Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Url:
    initial_url: str
    final_url: str
    title: str
    source: str
    isAI: bool
    created_at: Optional[datetime]
    id: Optional[int] = None  # Auto-increment primary key

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the urls table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initial_url TEXT NOT NULL UNIQUE,
                final_url TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                isAI BOOLEAN,
                created_at TEXT
            )
        """)
        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this URL record into the database"""
        cursor = conn.execute("""
            INSERT INTO urls (initial_url, final_url, title, source, isAI, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.initial_url, self.final_url, self.title, self.source, self.isAI,
              self.created_at.isoformat() if self.created_at else None))
        self.id = cursor.lastrowid  # Capture auto-generated id
        conn.commit()

    def update(self, conn: sqlite3.Connection):
        """Update this URL record in the database"""
        conn.execute("""
            UPDATE urls SET final_url = ?, title = ?, source = ?, isAI = ?, created_at = ?
            WHERE initial_url = ?
        """, (self.final_url, self.title, self.source, self.isAI,
              self.created_at.isoformat() if self.created_at else None,
              self.initial_url))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        """Delete this URL record from the database"""
        conn.execute("DELETE FROM urls WHERE initial_url = ?",
                     (self.initial_url,))
        conn.commit()

    @classmethod
    def get(cls, conn: sqlite3.Connection, initial_url: str) -> Optional['Url']:
        """Get a URL record by initial_url"""
        cursor = conn.execute("""
            SELECT id, initial_url, final_url, title, source, isAI, created_at
            FROM urls WHERE initial_url = ?
        """, (initial_url,))
        row = cursor.fetchone()
        if row:
            return cls(
                initial_url=row[1],
                final_url=row[2],
                title=row[3],
                source=row[4],
                isAI=bool(row[5]),
                created_at=datetime.fromisoformat(row[6]) if row[6] else None,
                id=row[0]
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Url']:
        """Get all URL records"""
        cursor = conn.execute("""
            SELECT id, initial_url, final_url, title, source, isAI, created_at FROM urls
        """)
        rows = cursor.fetchall()
        return [cls(
            initial_url=row[1],
            final_url=row[2],
            title=row[3],
            source=row[4],
            isAI=bool(row[5]),
            created_at=datetime.fromisoformat(row[6]) if row[6] else None,
            id=row[0]
        ) for row in rows]

    @classmethod
    def get_by_source_and_title(cls, conn: sqlite3.Connection, source: str, title: str) -> Optional['Url']:
        """Get a URL record by matching both source and title"""
        cursor = conn.execute("""
            SELECT id, initial_url, final_url, title, source, isAI, created_at
            FROM urls WHERE source = ? AND title = ?
        """, (source, title))
        row = cursor.fetchone()
        if row:
            return cls(
                initial_url=row[1],
                final_url=row[2],
                title=row[3],
                source=row[4],
                isAI=bool(row[5]),
                created_at=datetime.fromisoformat(row[6]) if row[6] else None,
                id=row[0]
            )
        return None

    @classmethod
    def get_by_url_or_source_and_title(cls, conn: sqlite3.Connection, url: str, source: str, title: str) -> Optional['Url']:
        """Get a URL record by URL first, then fallback to source and title match"""
        # First try to get by URL
        result = cls.get(conn, url)
        if result:
            return result

        # If not found by URL, try by source and title
        return cls.get_by_source_and_title(conn, source, title)

    def upsert(self, conn: sqlite3.Connection):
        """Insert or update this URL record"""
        conn.execute("""
            INSERT INTO urls (initial_url, final_url, title, source, isAI, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(initial_url) DO UPDATE SET
                final_url = excluded.final_url,
                title = excluded.title,
                source = excluded.source,
                isAI = excluded.isAI,
                created_at = excluded.created_at
        """, (self.initial_url, self.final_url, self.title, self.source, self.isAI,
              self.created_at.isoformat() if self.created_at else None))
        conn.commit()


@dataclass
class Article:
    final_url: str
    url: str
    source: str
    title: str
    published: Optional[datetime]
    rss_summary: Optional[str]
    isAI: bool
    status: Optional[str]
    html_path: Optional[str]
    last_updated: Optional[datetime]
    text_path: Optional[str]
    content_length: int
    summary: Optional[str]
    short_summary: Optional[str]
    description: Optional[str]
    rating: float
    cluster_label: Optional[str]
    domain: str
    site_name: str
    reputation: Optional[float]
    date: Optional[datetime]
    id: Optional[int] = None  # Auto-increment primary key

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the articles table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                final_url TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                published TEXT,
                rss_summary TEXT,
                isAI BOOLEAN NOT NULL,
                status TEXT,
                html_path TEXT,
                last_updated TEXT,
                text_path TEXT,
                content_length INTEGER NOT NULL,
                summary TEXT,
                short_summary TEXT,
                description TEXT,
                rating REAL NOT NULL,
                cluster_label TEXT,
                domain TEXT NOT NULL,
                site_name TEXT NOT NULL,
                reputation REAL,
                date TEXT
            )
        """)
        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this Article record into the database"""
        cursor = conn.execute("""
            INSERT INTO articles (final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.final_url, self.url, self.source, self.title,
            self.published.isoformat() if self.published else None,
            self.rss_summary, self.isAI, self.status, self.html_path,
            self.last_updated.isoformat() if self.last_updated else None,
            self.text_path, self.content_length, self.summary, self.short_summary, self.description,
            self.rating, self.cluster_label, self.domain, self.site_name,
            self.reputation,
            self.date.isoformat() if self.date else None
        ))
        self.id = cursor.lastrowid  # Capture auto-generated id
        conn.commit()

    def update(self, conn: sqlite3.Connection):
        """Update this Article record in the database"""
        conn.execute("""
            UPDATE articles SET url = ?, source = ?, title = ?, published = ?, rss_summary = ?, isAI = ?, status = ?, html_path = ?, last_updated = ?, text_path = ?, content_length = ?, summary = ?, short_summary = ?, description = ?, rating = ?, cluster_label = ?, domain = ?, site_name = ?, reputation = ?, date = ?
            WHERE final_url = ?
        """, (
            self.url, self.source, self.title,
            self.published.isoformat() if self.published else None,
            self.rss_summary, self.isAI, self.status, self.html_path,
            self.last_updated.isoformat() if self.last_updated else None,
            self.text_path, self.content_length, self.summary, self.short_summary, self.description,
            self.rating, self.cluster_label, self.domain, self.site_name,
            self.reputation,
            self.date.isoformat() if self.date else None, self.final_url
        ))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        """Delete this Article record from the database"""
        conn.execute("DELETE FROM articles WHERE final_url = ?",
                     (self.final_url,))
        conn.commit()

    @classmethod
    def get(cls, conn: sqlite3.Connection, final_url: str) -> Optional['Article']:
        """Get an Article record by final_url"""
        cursor = conn.execute("""
            SELECT id, final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date
            FROM articles WHERE final_url = ?
        """, (final_url,))
        row = cursor.fetchone()
        if row:
            return cls(
                final_url=row[1],
                url=row[2],
                source=row[3],
                title=row[4],
                published=datetime.fromisoformat(row[5]) if row[5] else None,
                rss_summary=row[6],
                isAI=bool(row[7]),
                status=row[8],
                html_path=row[9],
                last_updated=datetime.fromisoformat(
                    row[10]) if row[10] else None,
                text_path=row[11],
                content_length=row[12],
                summary=row[13],
                short_summary=row[14],
                description=row[15],
                rating=row[16],
                cluster_label=row[17],
                domain=row[18],
                site_name=row[19],
                reputation=row[20],
                date=datetime.fromisoformat(row[21]) if row[21] else None,
                id=row[0]
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Article']:
        """Get all Article records"""
        cursor = conn.execute("""
            SELECT id, final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date FROM articles
        """)
        rows = cursor.fetchall()
        return [cls(
            final_url=row[1],
            url=row[2],
            source=row[3],
            title=row[4],
            published=datetime.fromisoformat(row[5]) if row[5] else None,
            rss_summary=row[6],
            isAI=bool(row[7]),
            status=row[8],
            html_path=row[9],
            last_updated=datetime.fromisoformat(row[10]) if row[10] else None,
            text_path=row[11],
            content_length=row[12],
            summary=row[13],
            short_summary=row[14],
            description=row[15],
            rating=row[16],
            cluster_label=row[17],
            domain=row[18],
            site_name=row[19],
            reputation=row[20],
            date=datetime.fromisoformat(row[21]) if row[21] else None,
            id=row[0]
        ) for row in rows]

    def upsert(self, conn: sqlite3.Connection):
        """Insert or update this Article record"""
        conn.execute("""
            INSERT INTO articles (final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(final_url) DO UPDATE SET
                url = excluded.url,
                source = excluded.source,
                title = excluded.title,
                published = excluded.published,
                rss_summary = excluded.rss_summary,
                isAI = excluded.isAI,
                status = excluded.status,
                html_path = excluded.html_path,
                last_updated = excluded.last_updated,
                text_path = excluded.text_path,
                content_length = excluded.content_length,
                summary = excluded.summary,
                short_summary = excluded.short_summary,
                description = excluded.description,
                rating = excluded.rating,
                cluster_label = excluded.cluster_label,
                domain = excluded.domain,
                site_name = excluded.site_name,
                reputation = excluded.reputation,
                date = excluded.date
        """, (
            self.final_url, self.url, self.source, self.title,
            self.published.isoformat() if self.published else None,
            self.rss_summary, self.isAI, self.status, self.html_path,
            self.last_updated.isoformat() if self.last_updated else None,
            self.text_path, self.content_length, self.summary, self.short_summary, self.description,
            self.rating, self.cluster_label, self.domain, self.site_name,
            self.reputation,
            self.date.isoformat() if self.date else None
        ))
        conn.commit()


@dataclass
class Site:
    domain_name: str
    site_name: str
    reputation: float
    id: Optional[int] = None  # Auto-increment primary key

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the sites table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_name TEXT NOT NULL UNIQUE,
                site_name TEXT NOT NULL,
                reputation REAL NOT NULL
            )
        """)
        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this Site record into the database"""
        cursor = conn.execute("""
            INSERT INTO sites (domain_name, site_name, reputation)
            VALUES (?, ?, ?)
        """, (self.domain_name, self.site_name, self.reputation))
        self.id = cursor.lastrowid  # Capture auto-generated id
        conn.commit()

    def update(self, conn: sqlite3.Connection):
        """Update this Site record in the database"""
        conn.execute("""
            UPDATE sites SET site_name = ?, reputation = ?
            WHERE domain_name = ?
        """, (self.site_name, self.reputation, self.domain_name))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        """Delete this Site record from the database"""
        conn.execute("DELETE FROM sites WHERE domain_name = ?",
                     (self.domain_name,))
        conn.commit()

    @classmethod
    def get(cls, conn: sqlite3.Connection, domain_name: str) -> Optional['Site']:
        """Get a Site record by domain_name"""
        cursor = conn.execute("""
            SELECT id, domain_name, site_name, reputation
            FROM sites WHERE domain_name = ?
        """, (domain_name,))
        row = cursor.fetchone()
        if row:
            return cls(
                domain_name=row[1],
                site_name=row[2],
                reputation=row[3],
                id=row[0]
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Site']:
        """Get all Site records"""
        cursor = conn.execute("""
            SELECT id, domain_name, site_name, reputation FROM sites
        """)
        rows = cursor.fetchall()
        return [cls(
            domain_name=row[1],
            site_name=row[2],
            reputation=row[3],
            id=row[0]
        ) for row in rows]

    def upsert(self, conn: sqlite3.Connection):
        """Insert or update this Site record"""
        conn.execute("""
            INSERT INTO sites (domain_name, site_name, reputation)
            VALUES (?, ?, ?)
            ON CONFLICT(domain_name) DO UPDATE SET
                site_name = excluded.site_name,
                reputation = excluded.reputation
        """, (self.domain_name, self.site_name, self.reputation))
        conn.commit()


@dataclass
class Newsletter:
    session_id: str
    date: datetime
    final_newsletter: str
    id: Optional[int] = None  # Auto-increment primary key

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the newsletters table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS newsletters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                date TEXT NOT NULL,
                final_newsletter TEXT NOT NULL
            )
        """)
        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this Newsletter record into the database"""
        cursor = conn.execute("""
            INSERT INTO newsletters (session_id, date, final_newsletter)
            VALUES (?, ?, ?)
        """, (self.session_id, self.date.isoformat(), self.final_newsletter))
        self.id = cursor.lastrowid  # Capture auto-generated id
        conn.commit()

    def update(self, conn: sqlite3.Connection):
        """Update this Newsletter record in the database"""
        conn.execute("""
            UPDATE newsletters SET date = ?, final_newsletter = ?
            WHERE session_id = ?
        """, (self.date.isoformat(), self.final_newsletter, self.session_id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        """Delete this Newsletter record from the database"""
        conn.execute("DELETE FROM newsletters WHERE session_id = ?",
                     (self.session_id,))
        conn.commit()

    @classmethod
    def get(cls, conn: sqlite3.Connection, session_id: str) -> Optional['Newsletter']:
        """Get a Newsletter record by session_id"""
        cursor = conn.execute("""
            SELECT id, session_id, date, final_newsletter
            FROM newsletters WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            return cls(
                session_id=row[1],
                date=datetime.fromisoformat(row[2]),
                final_newsletter=row[3],
                id=row[0]
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Newsletter']:
        """Get all Newsletter records"""
        cursor = conn.execute("""
            SELECT id, session_id, date, final_newsletter FROM newsletters
        """)
        rows = cursor.fetchall()
        return [cls(
            session_id=row[1],
            date=datetime.fromisoformat(row[2]),
            final_newsletter=row[3],
            id=row[0]
        ) for row in rows]

    def upsert(self, conn: sqlite3.Connection):
        """Insert or update this Newsletter record"""
        conn.execute("""
            INSERT INTO newsletters (session_id, date, final_newsletter)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                date = excluded.date,
                final_newsletter = excluded.final_newsletter
        """, (self.session_id, self.date.isoformat(), self.final_newsletter))
        conn.commit()


@dataclass
class AgentState:
    session_id: str
    step_name: str
    state_data: str
    updated_at: datetime
    id: Optional[int] = None  # Auto-increment primary key

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the agent_state table with automatic migration from old schema"""
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                state_data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, step_name)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_state_session_id
            ON agent_state(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_state_updated_at
            ON agent_state(updated_at)
        """)

        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this AgentState record into the database"""
        cursor = conn.execute("""
            INSERT INTO agent_state (session_id, step_name, state_data, updated_at)
            VALUES (?, ?, ?, ?)
        """, (self.session_id, self.step_name, self.state_data,
              self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at))
        self.id = cursor.lastrowid  # Capture auto-generated id
        conn.commit()

    def update(self, conn: sqlite3.Connection):
        """Update this AgentState record in the database by id"""
        conn.execute("""
            UPDATE agent_state
            SET session_id = ?, step_name = ?, state_data = ?, updated_at = ?
            WHERE id = ?
        """, (self.session_id, self.step_name, self.state_data,
              self.updated_at.isoformat() if isinstance(
                  self.updated_at, datetime) else self.updated_at,
              self.id))
        conn.commit()

    def delete(self, conn: sqlite3.Connection):
        """Delete this AgentState record from the database"""
        conn.execute("DELETE FROM agent_state WHERE id = ?", (self.id,))
        conn.commit()

    def upsert(self, conn: sqlite3.Connection):
        """Insert or update this AgentState record based on UNIQUE(session_id, step_name)"""
        cursor = conn.execute("""
            INSERT INTO agent_state (session_id, step_name, state_data, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, step_name) DO UPDATE SET
                state_data = excluded.state_data,
                updated_at = excluded.updated_at
        """, (self.session_id, self.step_name, self.state_data,
              self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at))

        # If this was an insert, capture the new id
        if cursor.lastrowid:
            self.id = cursor.lastrowid
        else:
            # Was an update, fetch the existing id
            result = conn.execute("""
                SELECT id FROM agent_state
                WHERE session_id = ? AND step_name = ?
            """, (self.session_id, self.step_name)).fetchone()
            if result:
                self.id = result[0]

        conn.commit()

    @classmethod
    def get(cls, conn: sqlite3.Connection, record_id: int) -> Optional['AgentState']:
        """Get an AgentState record by id"""
        cursor = conn.execute("""
            SELECT id, session_id, step_name, state_data, updated_at
            FROM agent_state WHERE id = ?
        """, (record_id,))
        row = cursor.fetchone()
        if row:
            return cls(
                id=row[0],
                session_id=row[1],
                step_name=row[2],
                state_data=row[3],
                updated_at=datetime.fromisoformat(row[4]) if row[4] else None
            )
        return None

    @classmethod
    def get_by_session_and_step(cls, conn: sqlite3.Connection, session_id: str, step_name: str) -> Optional['AgentState']:
        """Get an AgentState record by session_id and step_name"""
        cursor = conn.execute("""
            SELECT id, session_id, step_name, state_data, updated_at
            FROM agent_state
            WHERE session_id = ? AND step_name = ?
        """, (session_id, step_name))
        row = cursor.fetchone()
        if row:
            return cls(
                id=row[0],
                session_id=row[1],
                step_name=row[2],
                state_data=row[3],
                updated_at=datetime.fromisoformat(row[4]) if row[4] else None
            )
        return None

    @classmethod
    def get_all_by_session(cls, conn: sqlite3.Connection, session_id: str) -> list['AgentState']:
        """Get all AgentState records for a session, ordered by updated_at ASC"""
        cursor = conn.execute("""
            SELECT id, session_id, step_name, state_data, updated_at
            FROM agent_state
            WHERE session_id = ?
            ORDER BY updated_at ASC
        """, (session_id,))
        rows = cursor.fetchall()
        return [cls(
            id=row[0],
            session_id=row[1],
            step_name=row[2],
            state_data=row[3],
            updated_at=datetime.fromisoformat(row[4]) if row[4] else None
        ) for row in rows]

    @classmethod
    def get_latest_by_session(cls, conn: sqlite3.Connection, session_id: str) -> Optional['AgentState']:
        """Get the most recent AgentState record for a session"""
        cursor = conn.execute("""
            SELECT id, session_id, step_name, state_data, updated_at
            FROM agent_state
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            return cls(
                id=row[0],
                session_id=row[1],
                step_name=row[2],
                state_data=row[3],
                updated_at=datetime.fromisoformat(row[4]) if row[4] else None
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['AgentState']:
        """Get all AgentState records across all sessions"""
        cursor = conn.execute("""
            SELECT id, session_id, step_name, state_data, updated_at
            FROM agent_state
            ORDER BY updated_at DESC
        """)
        rows = cursor.fetchall()
        return [cls(
            id=row[0],
            session_id=row[1],
            step_name=row[2],
            state_data=row[3],
            updated_at=datetime.fromisoformat(row[4]) if row[4] else None
        ) for row in rows]

    @classmethod
    def list_sessions(cls, conn: sqlite3.Connection, updated_at: Optional[datetime] = None, n_records: Optional[int] = 10) -> list[str]:
        """
        Get list of unique session IDs ordered by descending id (newest first).

        Args:
            conn: Database connection
            updated_at: If provided, only include sessions with states updated after this timestamp
            n_records: If provided, limit results to first N sessions

        Returns:
            List of session_id strings ordered by newest first
        """
        # Build query dynamically based on parameters
        query = """
            SELECT DISTINCT session_id
            FROM agent_state
        """

        params = []

        if updated_at is not None:
            query += " WHERE updated_at > ?"
            params.append(updated_at.isoformat() if isinstance(
                updated_at, datetime) else updated_at)

        # Order by id descending to get newest sessions first
        query += " ORDER BY updated_at DESC"

        if n_records is not None:
            query += " LIMIT ?"
            params.append(n_records)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    @classmethod
    def delete_session(cls, conn: sqlite3.Connection, session_id: str) -> int:
        """
        Delete all AgentState records for a session.

        Args:
            conn: Database connection
            session_id: Session ID to delete

        Returns:
            Number of records deleted
        """
        cursor = conn.execute("""
            DELETE FROM agent_state WHERE session_id = ?
        """, (session_id,))
        conn.commit()
        return cursor.rowcount
