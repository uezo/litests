from abc import ABC, abstractmethod
import asyncio
import logging
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
        self.queue: asyncio.Queue[STSResponse] = asyncio.Queue()
        self.worker_task = None

    async def handle_response(self, response: STSResponse):
        await self.queue.put(response)

    async def stop_response(self, context_id: str):
        await self.cancel()

    @abstractmethod
    async def process_response_item(self, response: STSResponse):
        pass

    async def worker(self):
        while True:
            response = await self.queue.get()
            try:
                await self.process_response_item(response)

            except Exception as ex:
                logger.error(f"Error processing STSResponse: {ex}", exc_info=True)

            finally:
                self.queue.task_done()

    async def start(self):
        if not self.worker_task:
            await self.worker()

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            await asyncio.gather(self.worker_task, return_exceptions=True)

    async def cancel(self):
        while not self.queue.empty():
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
