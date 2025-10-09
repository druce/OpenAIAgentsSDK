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

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the urls table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                initial_url TEXT PRIMARY KEY,
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
        conn.execute("""
            INSERT INTO urls (initial_url, final_url, title, source, isAI, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.initial_url, self.final_url, self.title, self.source, self.isAI,
              self.created_at.isoformat() if self.created_at else None))
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
            SELECT initial_url, final_url, title, source, isAI, created_at
            FROM urls WHERE initial_url = ?
        """, (initial_url,))
        row = cursor.fetchone()
        if row:
            return cls(
                initial_url=row[0],
                final_url=row[1],
                title=row[2],
                source=row[3],
                isAI=bool(row[4]),
                created_at=datetime.fromisoformat(row[5]) if row[5] else None
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Url']:
        """Get all URL records"""
        cursor = conn.execute("""
            SELECT initial_url, final_url, title, source, isAI, created_at FROM urls
        """)
        rows = cursor.fetchall()
        return [cls(
            initial_url=row[0],
            final_url=row[1],
            title=row[2],
            source=row[3],
            isAI=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]) if row[5] else None
        ) for row in rows]

    @classmethod
    def get_by_source_and_title(cls, conn: sqlite3.Connection, source: str, title: str) -> Optional['Url']:
        """Get a URL record by matching both source and title"""
        cursor = conn.execute("""
            SELECT initial_url, final_url, title, source, isAI, created_at
            FROM urls WHERE source = ? AND title = ?
        """, (source, title))
        row = cursor.fetchone()
        if row:
            return cls(
                initial_url=row[0],
                final_url=row[1],
                title=row[2],
                source=row[3],
                isAI=bool(row[4]),
                created_at=datetime.fromisoformat(row[5]) if row[5] else None
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

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the articles table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                final_url TEXT PRIMARY KEY,
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
        conn.execute("""
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
            SELECT final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date
            FROM articles WHERE final_url = ?
        """, (final_url,))
        row = cursor.fetchone()
        if row:
            return cls(
                final_url=row[0],
                url=row[1],
                source=row[2],
                title=row[3],
                published=datetime.fromisoformat(row[4]) if row[4] else None,
                rss_summary=row[5],
                isAI=bool(row[6]),
                status=row[7],
                html_path=row[8],
                last_updated=datetime.fromisoformat(
                    row[9]) if row[9] else None,
                text_path=row[10],
                content_length=row[11],
                summary=row[12],
                short_summary=row[13],
                description=row[14],
                rating=row[15],
                cluster_label=row[16],
                domain=row[17],
                site_name=row[18],
                reputation=row[19],
                date=datetime.fromisoformat(row[20]) if row[20] else None
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Article']:
        """Get all Article records"""
        cursor = conn.execute("""
            SELECT final_url, url, source, title, published, rss_summary, isAI, status, html_path, last_updated, text_path, content_length, summary, short_summary, description, rating, cluster_label, domain, site_name, reputation, date FROM articles
        """)
        rows = cursor.fetchall()
        return [cls(
            final_url=row[0],
            url=row[1],
            source=row[2],
            title=row[3],
            published=datetime.fromisoformat(row[4]) if row[4] else None,
            rss_summary=row[5],
            isAI=bool(row[6]),
            status=row[7],
            html_path=row[8],
            last_updated=datetime.fromisoformat(row[9]) if row[9] else None,
            text_path=row[10],
            content_length=row[11],
            summary=row[12],
            short_summary=row[13],
            description=row[14],
            rating=row[15],
            cluster_label=row[16],
            domain=row[17],
            site_name=row[18],
            reputation=row[19],
            date=datetime.fromisoformat(row[20]) if row[20] else None
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

    @classmethod
    def create_table(cls, conn: sqlite3.Connection):
        """Create the sites table if it doesn't exist"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sites (
                domain_name TEXT PRIMARY KEY,
                site_name TEXT NOT NULL,
                reputation REAL NOT NULL
            )
        """)
        conn.commit()

    def insert(self, conn: sqlite3.Connection):
        """Insert this Site record into the database"""
        conn.execute("""
            INSERT INTO sites (domain_name, site_name, reputation)
            VALUES (?, ?, ?)
        """, (self.domain_name, self.site_name, self.reputation))
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
            SELECT domain_name, site_name, reputation
            FROM sites WHERE domain_name = ?
        """, (domain_name,))
        row = cursor.fetchone()
        if row:
            return cls(
                domain_name=row[0],
                site_name=row[1],
                reputation=row[2]
            )
        return None

    @classmethod
    def get_all(cls, conn: sqlite3.Connection) -> list['Site']:
        """Get all Site records"""
        cursor = conn.execute("""
            SELECT domain_name, site_name, reputation FROM sites
        """)
        rows = cursor.fetchall()
        return [cls(
            domain_name=row[0],
            site_name=row[1],
            reputation=row[2]
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
