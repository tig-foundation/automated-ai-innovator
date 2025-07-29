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


class EndpointType(Enum):
    """
    Enum for supported endpoint types
    """
    RESPONSES = "responses"
    COMPLETIONS = "completions"


class LLM:
    """
    Base class for LLM interaction with streaming support
    """
    def __init__(self, provider: LLMProvider, api_key: str, model: str, base_url: str = None, endpoint_type: EndpointType = EndpointType.RESPONSES):
        if provider == LLMProvider.OPENAI:
            if endpoint_type == EndpointType.RESPONSES:
                self.api_url = base_url or "https://api.openai.com/v1/responses"
            elif endpoint_type == EndpointType.COMPLETIONS:
                self.api_url = base_url or "https://api.openai.com/v1/chat/completions"
            else:
                raise ValueError(f"Unsupported endpoint type: {endpoint_type}")
        elif provider == LLMProvider.AKASH:
            self.api_url = base_url or "https://chatapi.akash.network/api/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: OpenAI, Akash.")
        
        self.model = model
        self.api_key = api_key
        self.endpoint_type = endpoint_type

    def send_prompt(self, **kwargs) -> Union[dict, Generator[dict, None, None]]:
        """
        Send a prompt to the LLM with optional parameters
        Args:
            input (str or list): The prompt to send to the LLM.
            stream (bool): Whether to stream the response. Defaults to False.
            temperature (float): Sampling temperature for the LLM. Defaults to 0.7.
            **kwargs: Additional parameters for the LLM API.

        Returns:
            dict: Complete response if stream=False
            Generator[dict]: Stream of response chunks if stream=True
        """
        
        if self.endpoint_type == EndpointType.RESPONSES:
            payload = self._create_responses_payload(**kwargs)
        else:  # COMPLETIONS
            payload = self._create_completions_payload(**kwargs)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            if kwargs.get('stream', False):
                if self.endpoint_type == EndpointType.RESPONSES:
                    return self._handle_streaming_response_complete(payload, headers)
                else:  # COMPLETIONS
                    return self._handle_completions_streaming_response_complete(payload, headers)
            else:
                return self._handle_non_streaming_response(payload, headers)
                
        except Exception as e:
            raise RuntimeError(f"Failed to send prompt: {str(e)}") from e

    def _create_responses_payload(self, **kwargs) -> dict:
        """Create payload for responses endpoint"""
        payload = {
            "model": self.model,
            **kwargs
        }
        
        return payload

    def _create_completions_payload(self, **kwargs) -> dict:
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

    def _handle_streaming_response(self, payload: dict, headers: dict):
        """Handle streaming response with Server-Sent Events for responses endpoint"""
        response = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"LLM API error: {error_message}")
        
        # Parse Server-Sent Events
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    # Remove 'data: ' prefix and parse JSON
                    json_str = line[6:]  # Remove 'data: '
                    if json_str.strip():  # Skip empty lines
                        data = json.loads(json_str)
                        yield data
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue

    def _handle_completions_streaming_response(self, payload: dict, headers: dict):
        """Handle streaming response for completions endpoint"""
        response = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"LLM API error: {error_message}")
        
        # Parse Server-Sent Events for chat completions format
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    json_str = line[6:]  # Remove 'data: '
                    if json_str.strip() and json_str.strip() != '[DONE]':
                        data = json.loads(json_str)
                        yield data
                except json.JSONDecodeError:
                    continue

    def _handle_streaming_response_complete(self, payload: dict, headers: dict) -> dict:
        """Handle streaming response and return complete response dict for responses endpoint"""
        complete_text = ""
        response_id = None
        
        for chunk in self._handle_streaming_response(payload, headers):
            if chunk.get("type") == "response.output_text.delta":
                complete_text += chunk.get("delta", "")
            elif chunk.get("type") == "response.started":
                response_id = chunk.get("response", {}).get("id")
        
        # Return response in the same format as non-streaming
        return {
            "id": response_id,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": complete_text
                        }
                    ]
                }
            ]
        }

    def _handle_completions_streaming_response_complete(self, payload: dict, headers: dict) -> dict:
        """Handle streaming response and return complete response dict for completions endpoint"""
        complete_text = ""
        response_id = None
        model = payload.get("model", "")
        
        for chunk in self._handle_completions_streaming_response(payload, headers):
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    complete_text += choice["delta"]["content"]
                if "id" not in locals() or not response_id:
                    response_id = chunk.get("id")
        
        # Return response in chat completions format
        return {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": complete_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }