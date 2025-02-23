import logging
from fastapi import FastAPI
from litests import LiteSTS
from litests.vad import SpeechDetectorDummy
from litests.adapter.http import HttpAdapter

OPENAI_API_KEY = "YOUR_API_KEY"
GOOGLE_API_KEY = "YOUR_API_KEY"

logger = logging.getLogger(__name__)

# Create pipeline
sts = LiteSTS(
    vad=SpeechDetectorDummy(),  # Disable VAD
    stt_google_api_key=GOOGLE_API_KEY,
    llm_openai_api_key=OPENAI_API_KEY,
    debug=True
)

# Set HTTP adapter
adapter = HttpAdapter(sts)
router = adapter.get_api_router()

# Start HTTP server
app = FastAPI()
app.include_router(router)
