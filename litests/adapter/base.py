from abc import ABC, abstractmethod
from ..models import STSResponse
from ..pipeline import LiteSTS


class Adapter(ABC):
    def __init__(self, sts: LiteSTS):
        self.sts = sts
        self.sts.handle_response = self.handle_response
        self.sts.stop_response = self.stop_response

    @abstractmethod
    async def handle_response(self, response: STSResponse):
        pass

    @abstractmethod
    async def stop_response(self, context_id: str):
        pass
