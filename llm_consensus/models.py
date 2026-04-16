"""Model wrappers for OpenAI, Anthropic, and Google Gemini."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelResponse:
    model_id: str
    provider: str
    content: str
    round_num: int

class OpenAIModel:
    def __init__(self, model: str = "gpt-4o", instance_id: int = 1):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.instance_id = instance_id
        self.model_id = f"{model}-instance-{instance_id}"

    def ask(self, messages: list[dict], round_num: int = 0) -> ModelResponse:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.3)
        return ModelResponse(self.model_id, "openai", resp.choices[0].message.content.strip(), round_num)

class AnthropicModel:
    def __init__(self, model: str = "claude-sonnet-4-6", instance_id: int = 1):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.instance_id = instance_id
        self.model_id = f"claude-{model}-instance-{instance_id}"

    def ask(self, messages: list[dict], round_num: int = 0) -> ModelResponse:
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        kwargs = dict(model=self.model, max_tokens=2048, messages=user_msgs)
        if system_msgs:
            kwargs["system"] = system_msgs[0]
        resp = self.client.messages.create(**kwargs)
        return ModelResponse(self.model_id, "anthropic", resp.content[0].text.strip(), round_num)

class GeminiModel:
    def __init__(self, model: str = "gemini-1.5-pro", instance_id: int = 1):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.instance_id = instance_id
        self.model_id = f"gemini-{model}-instance-{instance_id}"

    def ask(self, messages: list[dict], round_num: int = 0) -> ModelResponse:
        parts = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)
        resp = self.client.generate_content(parts)
        return ModelResponse(self.model_id, "gemini", resp.text.strip(), round_num)

def build_default_pool(openai_instances=2, anthropic_instances=2, gemini_instances=2):
    pool = []
    for i in range(1, openai_instances + 1): pool.append(OpenAIModel(instance_id=i))
    for i in range(1, anthropic_instances + 1): pool.append(AnthropicModel(instance_id=i))
    for i in range(1, gemini_instances + 1): pool.append(GeminiModel(instance_id=i))
    return pool
