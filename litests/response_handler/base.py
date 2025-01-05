from abc import ABC, abstractmethod
import asyncio
import logging
import queue
import threading
from ..models import STSResponse

logger = logging.getLogger(__name__)


class ResponseHandler(ABC):
    def __init__(self):
        self.is_playing_locally = False

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, context_id: str):
        pass


class ResponseHandlerWithQueue(ResponseHandler):
    def __init__(self):
        super().__init__()
        self.queue: queue.Queue[STSResponse] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self.start_worker, daemon=True)
        self.worker_thread.start()

    async def handle_response(self, response: STSResponse):
        self.queue.put(response)

    async def stop_response(self, context_id: str):
        self.cancel()

    @abstractmethod
    async def process_response_item(self, response: STSResponse):
        pass

    def start_worker(self):
        while True:
            response = self.queue.get()
            try:
                asyncio.run(self.process_response_item(response))

            except Exception as ex:
                logger.error(f"Error processing STSResponse: {ex}", exc_info=True)

            finally:
                self.queue.task_done()

    def close(self):
        self.stop_event.set()
        self.cancel()
        self.worker_thread.join()

    def cancel(self):
        while not self.queue.empty():
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except:
                break
