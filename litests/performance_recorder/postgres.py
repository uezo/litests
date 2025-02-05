from dataclasses import fields
from datetime import datetime, timezone
import logging
import queue
import threading
import time
import psycopg2
from . import PerformanceRecorder, PerformanceRecord

logger = logging.getLogger(__name__)


class PostgreSQLPerformanceRecorder(PerformanceRecorder):
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "litests",
        user: str = "postgres",
        password: str = None,
    ):
        self.connection_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
        self.record_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.init_db()

        self.worker_thread = threading.Thread(target=self.start_worker, daemon=True)
        self.worker_thread.start()

    def connect_db(self):
        return psycopg2.connect(**self.connection_params)

    def init_db(self):
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ,
                        context_id TEXT,
                        voice_length REAL,
                        stt_time REAL,
                        stop_response_time REAL,
                        llm_first_chunk_time REAL,
                        llm_first_voice_chunk_time REAL,
                        llm_time REAL,
                        tts_first_chunk_time REAL,
                        tts_time REAL,
                        total_time REAL,
                        stt_name TEXT,
                        llm_name TEXT,
                        tts_name TEXT,
                        request_text TEXT,
                        response_text TEXT,
                        response_voice_text TEXT
                    )
                    """
                )
            conn.commit()
        finally:
            conn.close()

    def start_worker(self):
        conn = self.connect_db()
        try:
            while not self.stop_event.is_set() or not self.record_queue.empty():
                try:
                    record = self.record_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    self.insert_record(conn, record)
                except (psycopg2.InterfaceError, psycopg2.OperationalError):
                    try:
                        conn.close()
                    except Exception:
                        pass

                    logger.warning("Connection is not available. Retrying insert_record with new connection...")
                    time.sleep(0.5)
                    conn = self.connect_db()
                    self.insert_record(conn, record)

                self.record_queue.task_done()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def insert_record(self, conn: psycopg2.extensions.connection, record: PerformanceRecord):
        columns = [field.name for field in fields(PerformanceRecord)] + ["created_at"]
        placeholders = ["%s"] * len(columns)
        values = [getattr(record, field.name) for field in fields(PerformanceRecord)] + [datetime.now(timezone.utc)]
        sql = f"INSERT INTO performance_records ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with conn.cursor() as cur:
            cur.execute(sql, values)
        conn.commit()

    def record(self, record: PerformanceRecord):
        self.record_queue.put(record)

    def close(self):
        self.stop_event.set()
        self.record_queue.join()
        self.worker_thread.join()
