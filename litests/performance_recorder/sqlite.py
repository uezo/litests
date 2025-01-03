from dataclasses import fields
from datetime import datetime, timezone
import queue
import sqlite3
import threading
from . import PerformanceRecorder, PerformanceRecord


class SQLitePerformanceRecorder(PerformanceRecorder):
    def __init__(self, db_path="performance.db"):
        self.db_path = db_path
        self.record_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.init_db()

        self.worker_thread = threading.Thread(target=self.start_worker, daemon=True)
        self.worker_thread.start()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP,
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
        finally:
            conn.close()

    def start_worker(self):
        conn = sqlite3.connect(self.db_path)
        try:
            while not self.stop_event.is_set() or not self.record_queue.empty():
                try:
                    record = self.record_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.insert_record(conn, record)
                self.record_queue.task_done()
        finally:
            conn.close()

    def insert_record(self, conn: sqlite3.Connection, record: PerformanceRecord):
        columns = [field.name for field in fields(PerformanceRecord)] + ["created_at"]
        placeholders = ["?"] * len(columns)
        values = [getattr(record, field.name) for field in fields(PerformanceRecord)] + [datetime.now(timezone.utc)]
        sql = f"INSERT INTO performance_records ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        conn.execute(sql, values)
        conn.commit()

    def record(self, record: PerformanceRecord):
        self.record_queue.put(record)

    def close(self):
        self.stop_event.set()
        self.record_queue.join()
        self.worker_thread.join()
