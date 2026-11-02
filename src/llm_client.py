
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class OllamaClient:
    """Communicate with a locally running Ollama language model."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> str:
        """Send messages to Ollama and return the generated response."""

        if not messages:
            raise ValueError("Messages cannot be empty")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))

            content = result.get("message", {}).get("content", "")

            if not content.strip():
                raise RuntimeError("Ollama returned an empty response")

            return content.strip()

        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama API error ({error.code}): {details}"
            ) from error

        except URLError as error:
            raise ConnectionError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is installed and running."
            ) from error

        except TimeoutError as error:
            raise TimeoutError(
                "Ollama request timed out. "
                "Try a smaller model or increase the timeout."
            ) from error

    def generate(
        self,
        prompt: str,
        system_prompt: str = (
            "You are a helpful AI assistant. "
            "Give clear and accurate answers."
        ),
    ) -> str:
        """Generate an answer from a simple text prompt."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self.chat(messages)
