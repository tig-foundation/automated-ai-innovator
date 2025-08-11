import requests
import json        
from enum import Enum


class LLMProvider(Enum):
    """
    Enum for supported LLM providers
    """
    OPENAI = "openai"
    AKASH = "akash"

class LLM:
    """
    Base class for LLM interaction with streaming support
    """
    def __init__(self, provider: LLMProvider, api_key: str, model: str, secret_messages: list):
        """
        Initialize the LLM with provider, API key, model, and secret messages.

        secret_messages is a list of dictionaries {role: str, content: str} that are appended to the prompt
        """
        if provider == LLMProvider.OPENAI:
            self.api_url = "https://api.openai.com/v1/chat/completions"
        elif provider == LLMProvider.AKASH:
            self.api_url = "https://chatapi.akash.network/api/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: OpenAI, Akash.")
        
        self.model = model
        self.api_key = api_key
        self.secret_messages = secret_messages

    def send_prompt(self, messages: list, **kwargs) -> dict:
        """
        Send a prompt to the LLM via the completions endpoint
        Refer to https://platform.openai.com/docs/api-reference/responses/create

        Args:
            **kwargs: Parameters for the completions endpoint.

        Returns:
            dict: Complete response
        """
        payload = {
            "model": self.model,
            "messages": self.secret_messages + messages,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            d = response.json()
            if response.status_code == 200:
                return d
            else:
                error_message = d.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"LLM API error: {error_message}")
        except Exception as e:
            raise RuntimeError(f"Failed to send prompt: {str(e)}") from e
