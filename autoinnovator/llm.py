import requests
import json        
from enum import Enum
from typing import Tuple, Optional, Union, Generator


class LLMProvider(Enum):
    """
    Enum for supported LLM providers
    """
    OPENAI = "openai"


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
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are: OpenAI.")
        
        self.model = model
        self.api_key = api_key
        self.endpoint_type = endpoint_type

    def send_prompt(self, input: Union[str, list], stream: bool = True, **kwargs) -> Union[dict, Generator[dict, None, None]]:
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
        
        # Handle instructions parameter (for compatibility with chat completion style)
        instructions = kwargs.pop('instructions', None)
        previous_response_id = kwargs.pop('previous_response_id', None)
        
        if self.endpoint_type == EndpointType.RESPONSES:
            payload = self._create_responses_payload(input, instructions, previous_response_id, stream, **kwargs)
        else:  # COMPLETIONS
            payload = self._create_completions_payload(input, instructions, stream, **kwargs)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            if stream:
                if self.endpoint_type == EndpointType.RESPONSES:
                    return self._handle_streaming_response_complete(payload, headers)
                else:  # COMPLETIONS
                    return self._handle_completions_streaming_response_complete(payload, headers)
            else:
                return self._handle_non_streaming_response(payload, headers)
                
        except Exception as e:
            raise RuntimeError(f"Failed to send prompt: {str(e)}") from e

    def _create_responses_payload(self, input: Union[str, list], instructions: str, previous_response_id: str, stream: bool, **kwargs) -> dict:
        """Create payload for responses endpoint"""
        # Convert string input to proper Responses API format
        if isinstance(input, str):
            # If instructions are provided, combine them with the input
            if instructions:
                combined_input = f"{instructions}\n\n{input}"
            else:
                combined_input = input
                
            formatted_input = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": combined_input
                        }
                    ]
                }
            ]
        else:
            formatted_input = input
        
        payload = {
            "model": self.model,
            "input": formatted_input,
            "stream": stream,
            **kwargs
        }
        
        # Add previous_response_id if provided
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
            
        return payload

    def _create_completions_payload(self, input: Union[str, list], instructions: str, stream: bool, **kwargs) -> dict:
        """Create payload for completions endpoint"""
        # Convert input to chat completions format
        if isinstance(input, str):
            messages = []
            
            # Add system message for instructions if provided
            if instructions:
                messages.append({
                    "role": "system",
                    "content": instructions
                })
            
            # Add user message
            messages.append({
                "role": "user", 
                "content": input
            })
        elif isinstance(input, list):
            # If input is already a list, assume it's in messages format or convert from responses format
            messages = []
            for item in input:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        # Convert from responses format
                        role = item.get("role", "user")
                        content_items = item.get("content", [])
                        
                        # Extract text from content items
                        text_content = ""
                        for content_item in content_items:
                            if content_item.get("type") == "input_text":
                                text_content += content_item.get("text", "")
                            elif content_item.get("type") == "text":
                                text_content += content_item.get("text", "")
                        
                        if text_content:
                            messages.append({
                                "role": role,
                                "content": text_content
                            })
                    elif "role" in item and "content" in item:
                        # Already in chat completions format
                        messages.append(item)
                        
            # Add system message for instructions if provided and no system message exists
            if instructions and not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": instructions
                })
        else:
            messages = [{"role": "user", "content": str(input)}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
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

    def get_complete_response(self, input: Union[str, list], **kwargs) -> str:
        """
        Get the complete text response (convenience method for streaming)
        Args:
            input (str or list): The prompt to send
            **kwargs: Additional parameters

        Returns:
            str: Complete response text
        """
        # Force streaming to get real-time response
        kwargs['stream'] = True
        
        complete_text = ""
        
        if self.endpoint_type == EndpointType.RESPONSES:
            for chunk in self.send_prompt(input, **kwargs):
                if chunk.get("type") == "response.output_text.delta":
                    complete_text += chunk.get("delta", "")
                elif chunk.get("type") == "response.completed":
                    break
        else:  # COMPLETIONS
            for chunk in self.send_prompt(input, **kwargs):
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        complete_text += choice["delta"]["content"]
        
        return complete_text

    def get_streaming_text(self, input: Union[str, list], **kwargs):
        """
        Get streaming text deltas (convenience method)
        Args:
            input (str or list): The prompt to send
            **kwargs: Additional parameters

        Yields:
            str: Text deltas as they arrive
        """
        kwargs['stream'] = True
        
        if self.endpoint_type == EndpointType.RESPONSES:
            for chunk in self.send_prompt(input, **kwargs):
                if chunk.get("type") == "response.output_text.delta":
                    delta = chunk.get("delta", "")
                    if delta:
                        yield delta
        else:  # COMPLETIONS
            for chunk in self.send_prompt(input, **kwargs):
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        delta = choice["delta"]["content"]
                        if delta:
                            yield delta

    def get_response_text(self, response_data: dict) -> str:
        """
        Extract text from a non-streaming response
        Args:
            response_data (dict): Response from send_prompt with stream=False

        Returns:
            str: Extracted text content
        """
        if self.endpoint_type == EndpointType.RESPONSES:
            if "output" in response_data and response_data["output"]:
                for output_item in response_data["output"]:
                    if output_item.get("type") == "message" and output_item.get("role") == "assistant":
                        content = output_item.get("content", [])
                        for content_item in content:
                            if content_item.get("type") == "output_text":
                                return content_item.get("text", "")
        else:  # COMPLETIONS
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                    
        return ""