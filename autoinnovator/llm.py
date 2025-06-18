import requests
import json        
from enum import Enum
from typing import Tuple, Optional


class LLMProvider(Enum):
    """
    Enum for supported LLM providers
    """
    OPENAI = "openai"


class LLM:
    """
    Base class for LLM interaction
    """
    def __init__(self, provider: LLMProvider, api_key: str, model: str):
        if provider == LLMProvider.OPENAI:
            self.api_url = "https://api.openai.com/v1/responses"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: OpenAI, Groq, NanoGPT.")
        self.model = model
        self.api_key = api_key

    def send_prompt(self, input: str, **kwargs) -> dict:
        """
        Send a prompt to the LLM with optional parameters
        Args:
            input (str): The prompt to send to the LLM.
            temperature (float): Sampling temperature for the LLM. Defaults to 0.7.
            **kwargs: Additional parameters for the LLM API. Check OpenAI documentation:
                https://platform.openai.com/docs/api-reference/responses/create

        Returns:
            The response content.
        """
        
        payload = {
            "model": self.model,
            "input": input,
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