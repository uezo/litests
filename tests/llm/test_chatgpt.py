import json
import os
import pytest
from typing import Any, Dict
from uuid import uuid4
from litests.llm.chatgpt import ChatGPTService, ToolCall

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

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
async def test_chatgpt_service_simple():
    """
    Test ChatGPTService with a basic prompt to check if it can stream responses.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
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
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_cot():
    """
    Test ChatGPTService with a prompt to check Chain-of-Thought.
    This test actually calls OpenAI API, so it may cost tokens.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT_COT,
        model=MODEL,
        temperature=0.5,
        skip_before="<answer>"
    )
    context_id = f"test_cot_context_{uuid4()}"

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
    assert any(m["role"] == "assistant" for m in messages), "Assistant message not found in context."

    await service.openai_client.close()


@pytest.mark.asyncio
async def test_chatgpt_service_tool_calls():
    """
    Test ChatGPTService with a registered tool.
    The conversation might trigger the tool call, then the tool's result is fed back.
    This is just an example. The actual trigger depends on the model response.
    """
    service = ChatGPTService(
        openai_api_key=OPENAI_API_KEY,
        system_prompt="You can call a tool to solve math problems if necessary.",
        model=MODEL,
        temperature=0.5
    )
    context_id = f"test_tool_context_{uuid4()}"

    # Tool
    async def solve_math(problem: str) -> Dict[str, Any]:
        """
        Tool function example: parse the problem and return a result.
        """
        if problem.strip() == "1+1":
            return {"answer": 2}
        else:
            return {"answer": "unknown"}

    # Register tool
    tool_spec = {
        "type": "function",
        "function": {
            "name": "solve_math",
            "description": "Solve simple math problems",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"}
                },
                "required": ["problem"]
            }
        }
    }
    service.register_tool(tool_spec, solve_math)

    async def on_before_tool_calls(tool_calls: list[ToolCall]):
        pass
    service.on_before_tool_calls = on_before_tool_calls

    user_message = "次の問題を解いて: 1+1"
    collected_text = []

    async for resp in service.chat_stream(context_id, user_message):
        collected_text.append(resp.text)

    # Check context
    messages = await service.context_manager.get_histories(context_id)
    assert len(messages) == 4

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == user_message

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"] is not None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "solve_math"
    tool_call_id = messages[1]["tool_calls"][0]["id"]

    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == tool_call_id
    assert messages[2]["content"] == json.dumps({"answer": 2})

    assert messages[3]["role"] == "assistant"
    assert "2" in messages[3]["content"]

    await service.openai_client.close()
