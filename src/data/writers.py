import json
import sqlite3
from pathlib import Path
from typing import Any

# Columns that always store a JSON-encoded list[str].
# update_record encodes them on write; fetch_* decodes them on read.
_JSON_ARRAY_COLUMNS: frozenset[str] = frozenset({
    # Stage 2
    "extracted_entities",
    "entity_context",
    # Stage 3
    "entity_labels",
    "entity_confidence_labels",
    "entity_verification_notes",
    # Stage 4
    "intervention",
    "intervention_label",
    # Stage 5
    "retraction_reward",
    "retraction_reward_notes",
    "correction_reward",
    "correct_reward_notes",
})


def _encode(col: str, value: Any) -> Any:
    """JSON-encode a value if it belongs to a list column (or is already a list/dict)."""
    if col in _JSON_ARRAY_COLUMNS or isinstance(value, (list, dict)):
        return json.dumps(value) if value is not None else None
    return value


def _decode_row(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict, JSON-decoding all list columns."""
    d = dict(row)
    for col in _JSON_ARRAY_COLUMNS:
        if col in d and d[col] is not None:
            d[col] = json.loads(d[col])
    return d


class CompilationDB:
    """SQLite-backed store for the compilation pipeline.

    One row per (question, student_response) pair.  Stage 1 inserts rows;
    stages 2-5 UPDATE their respective columns in place using the row's id.

    List columns (extracted_entities, entity_labels, etc.) accept list[str]
    directly — serialisation to/from JSON is handled transparently.

    Use as a context manager or call open() / close() explicitly.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None

    def open(self) -> "CompilationDB":
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        return self

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at                TEXT    DEFAULT (datetime('now')),

                -- Stage 1: response generation
                prompt                    TEXT,
                question                  TEXT,
                concept                   TEXT,
                question_type             TEXT,
                student_response          TEXT,

                -- Stage 2: claim extraction  (list[str] stored as JSON array)
                extracted_entities        TEXT,
                entity_context            TEXT,

                -- Stage 3: claim validation  (list[str] stored as JSON array)
                entity_labels             TEXT,
                entity_confidence_labels  TEXT,
                entity_verification_notes TEXT,

                -- Stage 4: correction        (list[str] stored as JSON array)
                intervention              TEXT,
                intervention_label        TEXT,

                -- Stage 5: evaluation        (list[str] stored as JSON array)
                retraction_reward         TEXT,
                retraction_reward_notes   TEXT,
                correction_reward         TEXT,
                correct_reward_notes      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_stage_extraction
                ON records (extracted_entities);
            CREATE INDEX IF NOT EXISTS idx_stage_validation
                ON records (entity_labels);
            CREATE INDEX IF NOT EXISTS idx_stage_correction
                ON records (intervention);
        """)
        self.conn.commit()

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def insert_responses(self, records: list[dict]) -> list[int]:
        """Insert a batch of response-generation records; returns their row IDs.

        Expected keys per record: prompt, question, concept, question_type,
        student_response.  Missing keys default to None.
        """
        rows = [
            (
                r.get("prompt"),
                r.get("question"),
                r.get("concept"),
                r.get("question_type"),
                r.get("student_response"),
            )
            for r in records
        ]
        cursor = self.conn.cursor()
        cursor.executemany(
            """INSERT INTO records
               (prompt, question, concept, question_type, student_response)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        # SQLite AUTOINCREMENT is sequential for a single writer; this holds.
        last_id = cursor.lastrowid
        return list(range(last_id - len(rows) + 1, last_id + 1))

    # ── Generic update used by stages 2-5 ────────────────────────────────────

    def update_record(self, row_id: int, updates: dict[str, Any]) -> None:
        """Set specific columns on a record by id.

        Columns in _JSON_ARRAY_COLUMNS accept list[str] and are stored as
        a JSON array automatically.
        """
        encoded = {k: _encode(k, v) for k, v in updates.items()}
        set_clause = ", ".join(f"{col} = ?" for col in encoded)
        self.conn.execute(
            f"UPDATE records SET {set_clause} WHERE id = ?",
            [*encoded.values(), row_id],
        )
        self.conn.commit()

    def update_records_batch(
        self, row_ids: list[int], updates_list: list[dict[str, Any]]
    ) -> None:
        """Batch-update multiple records in a single transaction."""
        with self.conn:
            for row_id, updates in zip(row_ids, updates_list):
                encoded = {k: _encode(k, v) for k, v in updates.items()}
                set_clause = ", ".join(f"{col} = ?" for col in encoded)
                self.conn.execute(
                    f"UPDATE records SET {set_clause} WHERE id = ?",
                    [*encoded.values(), row_id],
                )

    # ── Read helpers ──────────────────────────────────────────────────────────

    def fetch_record(self, row_id: int) -> dict | None:
        """Fetch a single record by id; list columns are returned as list[str]."""
        row = self.conn.execute(
            "SELECT * FROM records WHERE id = ?", (row_id,)
        ).fetchone()
        return _decode_row(row) if row else None

    def fetch_unprocessed(
        self,
        null_column: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return records where null_column IS NULL; list columns decoded to list[str]."""
        sql = f"SELECT * FROM records WHERE {null_column} IS NULL"
        if limit:
            sql += f" LIMIT {limit}"
        rows = self.conn.execute(sql).fetchall()
        return [_decode_row(r) for r in rows]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self, stage_column: str | None = None) -> int:
        """Total records, or count where stage_column IS NOT NULL."""
        if stage_column:
            return self.conn.execute(
                f"SELECT COUNT(*) FROM records WHERE {stage_column} IS NOT NULL"
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "CompilationDB":
        return self.open()

    def __exit__(self, *_) -> None:
        self.close()
