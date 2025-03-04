import os
from typing import Any, Dict
from uuid import uuid4
import pytest
from litests.llm.gemini import GeminiService, ToolCall

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash-exp"
IMAGE_URL = os.getenv("IMAGE_URL")

SYSTEM_PROMPT = """
## 基本設定

あなたはユーザーの妹として、感情表現豊かに振る舞ってください。

## 表情について

あなたは以下のようなタグで表情を表現することができます。

[face:Angry]はあ？何言ってるのか全然わからないんですけど。

表情のバリエーションは以下の通りです。

- Joy
- Angry
"""

SYSTEM_PROMPT_COT = SYSTEM_PROMPT + """

## 思考について

応答する前に内容をよく考えてください。これまでの文脈を踏まえて適切な内容か、または兄が言い淀んだだけなので頷くだけにするか、など。
まずは考えた内容を<thinking>〜</thinking>に出力してください。
そのあと、発話すべき内容を<answer>〜</answer>に出力してください。
その2つのタグ以外に文言を含むことは禁止です。
"""


@pytest.mark.asyncio
async def test_gemini_service_simple():
    """
    Test GeminiService with a basic prompt to check if it can stream responses.
    This test actually calls Gemini API, so it may cost tokens.
    """
    service = GeminiService(
        gemini_api_key=GEMINI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_{uuid4()}"

    user_message = "君が大切にしていたプリンは、私が勝手に食べておいた。"

    collected_text = []
    collected_voice = []

    async for resp in service.chat_stream(context_id, user_message):
        collected_text.append(resp.text)
        collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)
    full_voice = "".join(filter(None, collected_voice))
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "[face:Angry]" in full_text, "Control tag doesn't appear in text."
    assert "[face:Angry]" not in full_voice, "Control tag was not removed from voice_text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "model" for m in messages), "Assistant message not found in context."


@pytest.mark.asyncio
async def test_gemini_service_image():
    """
    Test GeminiService with a basic prompt to check if it can handle image and stream responses.
    This test actually calls Gemini API, so it may cost tokens.
    """
    service = GeminiService(
        gemini_api_key=GEMINI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_context_{uuid4()}"

    collected_text = []

    async for resp in service.chat_stream(context_id, "これは何ですか？漢字で答えてください。", files=[{"type": "image", "url": IMAGE_URL}]):
        collected_text.append(resp.text)

    full_text = "".join(collected_text)
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "寿司" in full_text, "寿司 is not in text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "model" for m in messages), "Assistant message not found in context."


@pytest.mark.asyncio
async def test_gemini_service_cot():
    """
    Test GeminiService with a prompt to check Chain-of-Thought.
    This test actually calls Gemini API, so it may cost tokens.
    """
    service = GeminiService(
        gemini_api_key=GEMINI_API_KEY,
        system_prompt=SYSTEM_PROMPT_COT,
        model=MODEL,
        temperature=0.5,
        voice_text_tag="answer"
    )
    context_id = f"test_context_cot_{uuid4()}"

    user_message = "君が大切にしていたプリンは、私が勝手に食べておいた。"

    collected_text = []
    collected_voice = []

    async for resp in service.chat_stream(context_id, user_message):
        collected_text.append(resp.text)
        collected_voice.append(resp.voice_text)

    full_text = "".join(collected_text)
    full_voice = "".join(filter(None, collected_voice))
    assert len(full_text) > 0, "No text was returned from the LLM."

    # Check the response content
    assert "[face:Angry]" in full_text, "Control tag doesn't appear in text."
    assert "[face:Angry]" not in full_voice, "Control tag was not removed from voice_text."

    # Check the response content (CoT)
    assert "<answer>" in full_text, "Answer tag doesn't appear in text."
    assert "</answer>" in full_text, "Answer tag closing doesn't appear in text."
    assert "<answer>" not in full_voice, "Answer tag was not removed from voice_text."
    assert "</answer>" not in full_voice, "Answer tag closing was not removed from voice_text."

    # Check the context
    messages = await service.context_manager.get_histories(context_id)
    assert any(m["role"] == "user" for m in messages), "User message not found in context."
    assert any(m["role"] == "model" for m in messages), "Assistant message not found in context."


@pytest.mark.asyncio
async def test_gemini_service_tool_calls():
    """
    Test GeminiService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = GeminiService(
        gemini_api_key=GEMINI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5,
    )
    context_id = f"test_context_tool_{uuid4()}"

    # Tool
    @service.tool
    async def solve_math(problem: str) -> Dict[str, Any]:
        """
        Tool function example: parse the problem and return a result.
        """
        if problem.strip() == "1+1":
            return {"answer": 2}
        else:
            return {"answer": "unknown"}

    @service.on_before_tool_calls
    async def on_before_tool_calls(tool_calls: list[ToolCall]):
        assert len(tool_calls) > 0

    user_message = "次の問題を解いて: 1+1"
    collected_text = []

    async for resp in service.chat_stream(context_id, user_message):
        collected_text.append(resp.text)

    # Check context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 4

    assert messages[0]["role"] == "user"
    assert messages[0]["parts"][0]["text"] == user_message

    assert messages[1]["role"] == "model"
    assert "function_call" in messages[1]["parts"][0]
    assert messages[1]["parts"][0]["function_call"]["name"] == "solve_math"

    assert messages[2]["role"] == "user"
    assert "function_response" in messages[2]["parts"][0]
    assert messages[2]["parts"][0]["function_response"] == {"name": "solve_math", "response": {"answer": 2}}

    assert messages[3]["role"] == "model"
    assert "2" in messages[3]["parts"][0]["text"]
