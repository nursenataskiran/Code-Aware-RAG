import os
from typing import Optional

import requests
from dotenv import load_dotenv


load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # optional attribution headers
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Code-Aware-RAG",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Log the actual payload values being sent
        prompt_len = len(prompt)
        print(f"[OpenRouter] model={self.model}  max_tokens={max_tokens}  prompt_chars={prompt_len}")

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code == 402:
            # Dump the full error response so we can see the real reason
            print(f"⚠️ OpenRouter 402 — raw response body:")
            print(response.text)
            print(f"  → model: {self.model}")
            print(f"  → max_tokens: {max_tokens}")
            print(f"  → prompt length: {prompt_len} chars")
            print(f"  → api_key: ...{self.api_key[-8:]}")
            return "[ERROR: OpenRouter 402]"

        if not response.ok:
            print(f"⚠️ OpenRouter {response.status_code} — raw response body:")
            print(response.text)

        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]