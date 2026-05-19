import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

# Configuration with sensible defaults; can be overridden using environment variables
DB_DIR = os.environ.get("ORTHANC_FEEDBACK_DB_DIR", "/var/lib/odelia-feedback")
DB_PATH = os.environ.get(
    "ORTHANC_FEEDBACK_DB_PATH", os.path.join(DB_DIR, "feedback.sqlite")
)
ENABLE_WAL = os.environ.get("ORTHANC_FEEDBACK_ENABLE_WAL", "1") not in (
    "0",
    "false",
    "False",
)
BUSY_TIMEOUT_MS = int(os.environ.get("ORTHANC_FEEDBACK_BUSY_TIMEOUT_MS", "8000"))
CHECKPOINT_INTERVAL_SEC = int(
    os.environ.get("ORTHANC_FEEDBACK_CHECKPOINT_INTERVAL_SEC", str(5 * 60))
)


_init_lock = threading.Lock()
_initialized = False
_checkpoint_thread_started = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _connect() -> sqlite3.Connection:
    # check_same_thread=False to allow usage from handler threads
    cx = sqlite3.connect(
        DB_PATH, timeout=BUSY_TIMEOUT_MS / 1000.0, check_same_thread=False
    )
    cx.row_factory = sqlite3.Row
    # Enforce foreign keys
    cx.execute("PRAGMA foreign_keys=ON;")
    # WAL and busy timeout
    if ENABLE_WAL:
        try:
            mode = cx.execute("PRAGMA journal_mode=WAL;")
            assert mode.lower() == "wal", f"journal_mode is {mode}"
            print(mode)
        except Exception as e:
            print(e)
    cx.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    return cx


def _run_ddl(cx: sqlite3.Connection) -> None:
    cx.executescript(
        """
        CREATE TABLE IF NOT EXISTS ai_result_ref (
          id INTEGER PRIMARY KEY,
          study_uid TEXT NOT NULL,
          model_name TEXT NOT NULL,
          model_version TEXT NOT NULL,
          result_ts TEXT NOT NULL,
          meta_json TEXT,
          UNIQUE (study_uid, model_name, model_version, result_ts)
        );

        -- Append-only event log for feedback submissions (initial and edits)
        CREATE TABLE IF NOT EXISTS feedback_event (
          id INTEGER PRIMARY KEY,
          ai_result_id INTEGER NOT NULL REFERENCES ai_result_ref(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL,
          verdict_L INTEGER NOT NULL CHECK (verdict_L IN (-1,0,1)),
          verdict_R INTEGER NOT NULL CHECK (verdict_R IN (-1,0,1)),
          submission_kind TEXT NOT NULL CHECK (submission_kind IN ('initial','edit')),
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );

        -- Prevent updates/deletes on events
        CREATE TRIGGER IF NOT EXISTS trg_feedback_event_block_update
        BEFORE UPDATE ON feedback_event
        BEGIN
          SELECT RAISE(ABORT, 'Feedback events are immutable; updates disabled.');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_feedback_event_block_delete
        BEFORE DELETE ON feedback_event
        BEGIN
          SELECT RAISE(ABORT, 'Feedback event deletion disabled.');
        END;

        -- Current per-user verdict for each AI result (latest event wins)
        CREATE VIEW IF NOT EXISTS v_feedback_current AS
        WITH ranked AS (
          SELECT
            e.*,
            ROW_NUMBER() OVER (
              PARTITION BY e.ai_result_id, e.user_id
              ORDER BY e.created_at DESC, e.id DESC
            ) AS rn
          FROM feedback_event e
        )
        SELECT * FROM ranked WHERE rn = 1;

        -- Denormalized current view
        CREATE VIEW IF NOT EXISTS v_feedback_denorm AS
        SELECT c.id, r.study_uid, r.model_name, r.model_version, r.result_ts,
               c.user_id, c.verdict_L, c.verdict_R, c.created_at, c.submission_kind
        FROM v_feedback_current c
        JOIN ai_result_ref r ON r.id = c.ai_result_id;

        -- History view
        CREATE VIEW IF NOT EXISTS v_feedback_history_denorm AS
        SELECT e.id, r.study_uid, r.model_name, r.model_version, r.result_ts,
               e.user_id, e.verdict_L, e.verdict_R, e.created_at, e.submission_kind
        FROM feedback_event e
        JOIN ai_result_ref r ON r.id = e.ai_result_id;

        CREATE INDEX IF NOT EXISTS idx_result_by_study_model_ts
          ON ai_result_ref(study_uid, model_name, model_version, result_ts);
        CREATE INDEX IF NOT EXISTS idx_event_result_user_time
          ON feedback_event(ai_result_id, user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_event_user
          ON feedback_event(user_id);
        """
    )


def initialize() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _ensure_dir(DB_DIR)
        cx = _connect()
        try:
            _run_ddl(cx)
        finally:
            cx.close()
        _initialized = True


def _checkpoint_worker() -> None:
    while True:
        try:
            time.sleep(CHECKPOINT_INTERVAL_SEC)
            cx = _connect()
            try:
                cx.execute("PRAGMA wal_checkpoint(PASSIVE);")
            finally:
                cx.close()
        except Exception:
            # Keep the daemon thread alive even if checkpoint fails
            continue


def start_checkpoint_thread() -> None:
    global _checkpoint_thread_started
    if _checkpoint_thread_started or not ENABLE_WAL:
        return
    t = threading.Thread(
        target=_checkpoint_worker, name="feedback-sqlite-checkpoint", daemon=True
    )
    t.start()
    _checkpoint_thread_started = True


def _get_or_create_ai_result_id(
    cx: sqlite3.Connection,
    study_uid: str,
    model_name: str,
    model_version: str,
    result_ts: str,
    meta_json: Optional[str],
) -> int:
    try:
        cx.execute(
            """
            INSERT INTO ai_result_ref(study_uid, model_name, model_version, result_ts, meta_json)
            VALUES(?,?,?,?,?)
            ON CONFLICT(study_uid, model_name, model_version, result_ts)
            DO UPDATE SET meta_json=COALESCE(excluded.meta_json, ai_result_ref.meta_json)
            """,
            (study_uid, model_name, model_version, result_ts, meta_json),
        )
    except sqlite3.IntegrityError:
        pass
    row = cx.execute(
        "SELECT id FROM ai_result_ref WHERE study_uid=? AND model_name=? AND model_version=? AND result_ts=?",
        (study_uid, model_name, model_version, result_ts),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to get or create ai_result_ref")
    return int(row[0])


class ConflictError(Exception):
    pass


def submit_feedback(p: Dict) -> Dict:
    initialize()
    cx = _connect()
    try:
        cx.execute("BEGIN IMMEDIATE")
        ai_id = _get_or_create_ai_result_id(
            cx,
            p["study_uid"],
            p["model_name"],
            p["model_version"],
            p["result_ts"],
            p.get("meta_json"),
        )
        # Determine if prior submission exists for this user/result
        has_prior = (
            cx.execute(
                "SELECT 1 FROM feedback_event WHERE ai_result_id=? AND user_id=? LIMIT 1",
                (ai_id, p["user_id"]),
            ).fetchone()
            is not None
        )
        if has_prior and not bool(p.get("edited")):
            cx.execute("ROLLBACK")
            raise ConflictError("Already submitted; use edit flow")

        submission_kind = "edit" if has_prior else "initial"

        cx.execute(
            """
            INSERT INTO feedback_event(ai_result_id, user_id, verdict_L, verdict_R, submission_kind)
            VALUES(?,?,?,?,?)
            """,
            (
                ai_id,
                p["user_id"],
                int(p["verdict_L"]),
                int(p["verdict_R"]),
                submission_kind,
            ),
        )
        row = cx.execute(
            "SELECT id, created_at FROM feedback_event WHERE rowid = last_insert_rowid()"
        ).fetchone()
        cx.execute("COMMIT")
        return {
            "id": int(row[0]),
            "study_uid": p["study_uid"],
            "model_name": p["model_name"],
            "model_version": p["model_version"],
            "result_ts": p["result_ts"],
            "user_id": p["user_id"],
            "verdict_L": int(p["verdict_L"]),
            "verdict_R": int(p["verdict_R"]),
            "submission_kind": submission_kind,
            "created_at": row[1],
        }
    finally:
        cx.close()


def get_result_id(
    study_uid: str, model_name: str, model_version: str, result_ts: str
) -> Optional[int]:
    initialize()
    cx = _connect()
    try:
        row = cx.execute(
            "SELECT id FROM ai_result_ref WHERE study_uid=? AND model_name=? AND model_version=? AND result_ts=?",
            (study_uid, model_name, model_version, result_ts),
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        cx.close()


def read_feedback(
    study_uid: str,
    model_name: str,
    model_version: str,
    result_ts: str,
    include_users: bool = False,
    include_history: bool = False,
) -> Dict:
    initialize()
    cx = _connect()
    try:
        rid = get_result_id(study_uid, model_name, model_version, result_ts)
        if rid is None:
            # No submissions yet; return zeros
            base = {
                "study_uid": study_uid,
                "model_name": model_name,
                "model_version": model_version,
                "result_ts": result_ts,
                "n_submissions": 0,
                "aggregate": {
                    "L": {"agree": 0, "unsure": 0, "disagree": 0},
                    "R": {"agree": 0, "unsure": 0, "disagree": 0},
                },
            }
            if include_users:
                base["users"] = []
            return base

        row = cx.execute(
            """
            SELECT
              SUM(verdict_L=1) AS L_agree,
              SUM(verdict_L=0) AS L_unsure,
              SUM(verdict_L=-1) AS L_disagree,
              SUM(verdict_R=1) AS R_agree,
              SUM(verdict_R=0) AS R_unsure,
              SUM(verdict_R=-1) AS R_disagree,
              COUNT(*) AS n
            FROM v_feedback_current
            WHERE ai_result_id = ?
            """,
            (rid,),
        ).fetchone()
        result = {
            "study_uid": study_uid,
            "model_name": model_name,
            "model_version": model_version,
            "result_ts": result_ts,
            "n_submissions": int(row[6] or 0),
            "aggregate": {
                "L": {
                    "agree": int(row[0] or 0),
                    "unsure": int(row[1] or 0),
                    "disagree": int(row[2] or 0),
                },
                "R": {
                    "agree": int(row[3] or 0),
                    "unsure": int(row[4] or 0),
                    "disagree": int(row[5] or 0),
                },
            },
        }
        if include_users:
            users = [
                dict(r)
                for r in cx.execute(
                    "SELECT user_id, verdict_L, verdict_R, created_at, submission_kind FROM v_feedback_current WHERE ai_result_id=? ORDER BY created_at ASC",
                    (rid,),
                ).fetchall()
            ]
            result["users"] = users
        if include_history:
            history = [
                dict(r)
                for r in cx.execute(
                    "SELECT user_id, verdict_L, verdict_R, created_at, submission_kind FROM feedback_event WHERE ai_result_id=? ORDER BY created_at ASC, id ASC",
                    (rid,),
                ).fetchall()
            ]
            result["history"] = history
        return result
    finally:
        cx.close()


def register_result(
    study_uid: str,
    model_name: str,
    model_version: str,
    result_ts: str,
    meta_json: Optional[str],
) -> Dict:
    initialize()
    cx = _connect()
    try:
        before_id = get_result_id(study_uid, model_name, model_version, result_ts)
        cx.execute("BEGIN IMMEDIATE")
        _get_or_create_ai_result_id(
            cx, study_uid, model_name, model_version, result_ts, meta_json
        )
        cx.execute("COMMIT")
        after_id = get_result_id(study_uid, model_name, model_version, result_ts)
        return {"created": before_id is None, "id": after_id}
    finally:
        cx.close()


def export_rows_ndjson(
    since: Optional[str] = None,
    until: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    scope: str = "history",
) -> Iterable[str]:
    initialize()
    cx = _connect()
    try:
        clauses = []
        args: List[str] = []
        if since:
            clauses.append("e.created_at >= ?")
            args.append(since)
        if until:
            clauses.append("e.created_at <= ?")
            args.append(until)
        if model_name:
            clauses.append("r.model_name = ?")
            args.append(model_name)
        if model_version:
            clauses.append("r.model_version = ?")
            args.append(model_version)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        if scope == "current":
            sql = f"""
                SELECT r.study_uid, r.model_name, r.model_version, r.result_ts,
                       c.user_id, c.verdict_L, c.verdict_R, c.created_at, c.submission_kind
                FROM v_feedback_current c
                JOIN ai_result_ref r ON r.id = c.ai_result_id
                {where.replace("e.", "c.")}
                ORDER BY c.created_at ASC
            """
        else:
            sql = f"""
                SELECT r.study_uid, r.model_name, r.model_version, r.result_ts,
                       e.user_id, e.verdict_L, e.verdict_R, e.created_at, e.submission_kind
                FROM feedback_event e
                JOIN ai_result_ref r ON r.id = e.ai_result_id
                {where}
                ORDER BY e.created_at ASC
            """
        for r in cx.execute(sql, args):
            # Manual JSON to avoid importing json here; caller will add newlines
            obj = {
                "study_uid": r[0],
                "model_name": r[1],
                "model_version": r[2],
                "result_ts": r[3],
                "user_id": r[4],
                "verdict_L": int(r[5]),
                "verdict_R": int(r[6]),
                "created_at": r[7],
                "submission_kind": r[8],
            }
            yield obj
    finally:
        cx.close()


def export_rows_csv(
    since: Optional[str] = None,
    until: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    scope: str = "history",
) -> Tuple[str, Iterable[Tuple]]:
    header = "study_uid,model_name,model_version,result_ts,user_id,verdict_L,verdict_R,created_at,submission_kind\n"
    initialize()
    cx = _connect()

    def _iter():
        try:
            clauses = []
            args: List[str] = []
            if since:
                clauses.append("e.created_at >= ?")
                args.append(since)
            if until:
                clauses.append("e.created_at <= ?")
                args.append(until)
            if model_name:
                clauses.append("r.model_name = ?")
                args.append(model_name)
            if model_version:
                clauses.append("r.model_version = ?")
                args.append(model_version)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            if scope == "current":
                sql = f"""
                    SELECT r.study_uid, r.model_name, r.model_version, r.result_ts,
                           c.user_id, c.verdict_L, c.verdict_R, c.created_at, c.submission_kind
                    FROM v_feedback_current c
                    JOIN ai_result_ref r ON r.id = c.ai_result_id
                    {where.replace("e.", "c.")}
                    ORDER BY c.created_at ASC
                """
            else:
                sql = f"""
                    SELECT r.study_uid, r.model_name, r.model_version, r.result_ts,
                           e.user_id, e.verdict_L, e.verdict_R, e.created_at, e.submission_kind
                    FROM feedback_event e
                    JOIN ai_result_ref r ON r.id = e.ai_result_id
                    {where}
                    ORDER BY e.created_at ASC
                """
            for r in cx.execute(sql, args):
                yield r
        finally:
            cx.close()

    return header, _iter()


def health() -> Dict:
    initialize()
    cx = _connect()
    try:
        wal_mode = None
        try:
            wal_mode = cx.execute("PRAGMA journal_mode;").fetchone()[0]
        except Exception:
            wal_mode = "unknown"
        return {
            "db_ready": True,
            "wal_mode": wal_mode == "wal" if isinstance(wal_mode, str) else False,
            "sqlite_version": sqlite3.sqlite_version,
            "path": DB_PATH,
        }
    finally:
        cx.close()


# Initialize at import time and start checkpoint thread
initialize()
start_checkpoint_thread()
