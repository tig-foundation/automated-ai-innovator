import requests
import json        
from enum import Enum
from typing import Tuple, Optional, Union, Generator


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
    def __init__(self, provider: LLMProvider, api_key: str, model: str, base_url: str = None):
        if not model:
            raise ValueError("Model is required")

        if not base_url:
            if provider == LLMProvider.OPENAI:
                self.api_url = "https://api.openai.com/v1/chat/completions"
            elif provider == LLMProvider.AKASH:
                self.api_url = "https://chatapi.akash.network/api/v1/chat/completions"
            else:
                raise ValueError(f"Unsupported provider: {provider}. Supported providers are: OpenAI, Akash.")
        else:
            self.api_url = base_url
        
        self.model = model
        self.api_key = api_key

    def send_prompt(self, **kwargs) -> dict:
        """
        Send a prompt to the LLM with optional parameters
        Args:
            input (str or list): The prompt to send to the LLM.
            stream (bool): Whether to stream the response. Defaults to False.
            temperature (float): Sampling temperature for the LLM. Defaults to 0.7.
            **kwargs: Additional parameters for the LLM API.

        Returns:
            dict: Complete response
        """
        

        payload = self._create_payload(**kwargs)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            return self._handle_non_streaming_response(payload, headers)
        except Exception as e:
            raise RuntimeError(f"Failed to send prompt: {str(e)}") from e

    def _create_payload(self, **kwargs) -> dict:
        """Create payload for completions endpoint"""
        
        payload = {
            "model": self.model,
            **kwargs
        }

        return payload

    def _handle_non_streaming_response(self, payload: dict, headers: dict) -> dict:
        """Handle non-streaming response"""
        response = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"LLM API error: {error_message}")
