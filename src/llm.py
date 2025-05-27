"""
LLM API and builder functions
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any

import copy
import os
import requests
import json


class BaseAPI:
    def __init__(self):
        self.api_key = ""
        self.api_url = ""
    
    @abstractmethod
    def send_prompt(
        self, 
        messages: List[Dict[str, Any]] = None,
        model: str = None, 
        temperature: float = 0.7,
        max_tokens: int = None, 
        top_p: float = None,
        frequency_penalty: float = None, 
        presence_penalty: float = None
    ) -> tuple:
        """
        Send a prompt to the API with messages array.
        
        Args:
            messages: List of message objects with structure:
                     [{"role": "system"|"user"|"assistant", "content": str}]
            model: Model to use for this request
            temperature: Controls randomness (0-2, higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Controls diversity via nucleus sampling
            frequency_penalty: Reduces repetition of token sequences
            presence_penalty: Reduces repetition of topics
            
        Returns:
            Tuple of (success: bool, response: str)
        """
        pass
    
    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
    
    def set_api_url(self, api_url: str) -> None:
        self.api_url = api_url

    def extract_code_from_response(self, response: str) -> str:
        code_blocks = []
        in_code_block = False
        current_block = []
        
        for line in response.split('\n'):
            if line.startswith('```'):
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks



class OpenAIAPI(BaseAPI):
    def __init__(self, provider="openai"):
        super().__init__()
        
        if provider.lower() == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        elif provider.lower() == "groq":
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.api_key = os.environ.get("GROQ_API_KEY", "")
        elif provider.lower() == "nanogpt":
            self.api_url = "https://nano-gpt.com/api/v1/chat/completions"
            self.api_key = os.environ.get("NANOGPT_API_KEY", "")
        else:
            self.api_url = ""
            self.api_key = ""
    
    def send_prompt(
        self, 
        messages: list = None,
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.7,
        max_tokens: int = None, 
        top_p: float = None,
        frequency_penalty: float = None, 
        presence_penalty: float = None
    ) -> tuple:
        if not self.api_key:
            return False, "API key not set"
        
        if not self.api_url:
            return False, "API URL not set"
        
        messages = messages or []
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if top_p is not None:
            payload["top_p"] = top_p
        
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        
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
            
            response_json = response.json()
            
            if response.status_code == 200:
                content = response_json["choices"][-1]["message"]["content"]
                return True, content
            else:
                error_message = response_json.get("error", {}).get("message", "Unknown error")
                return False, f"API Error: {error_message}"
                
        except Exception as e:
            return False, f"Request failed: {str(e)}"
        


class BaseLLM(ABC):
    """
    Base class for LLM interaction
    """
    def __init__(self, model: str, temperature: float):
        self.api = None
        self.model = model
        self.max_tokens = None
        self.temperature = temperature
        self.system_prompt = None

    def get_system_prompt(self) -> Optional[str]:
        """Return the system prompt for this model"""
        return self.system_prompt
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for this model"""
        self.system_prompt = system_prompt

    def set_api(self, api: BaseAPI) -> None:
        """Set the API interface"""
        self.api = api
    
    def copy(self) -> 'BaseLLM':
        """Create a copy of this model"""
        return copy.deepcopy(self)

    def send_prompt(
            self, 
            prompt: str,
            **kwargs
        ) -> Tuple[bool, Any]:
        """
        Send prompt to LLM with optional program constraints and history

        :return:
            Boolean success flag, message str
        """
        if not self.api:
            return False, "API not set"
        
        system_prompt = self.get_system_prompt()
        if not system_prompt:
            return False, "System prompt not set"
        
        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": prompt}, 
        ]
        
        return self.api.send_prompt(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )