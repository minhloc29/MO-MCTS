import logging
from typing import Optional
from .base import BaseClient

try:
    import google.generativeai as genai
except ImportError:
    genai = 'google.generativeai'

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):

    ClientClass = genai

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        """
        Gemini API expects a single prompt instead of a list of chat messages like OpenAI.
        We'll concatenate them here.
        """
        # Convert messages list to a single prompt string
        prompt_parts = []
        for m in messages:
            role = m['role']
            content = m['content']
            if role == 'system':
                prompt_parts.append(f"[System]: {content}")
            elif role == 'user':
                prompt_parts.append(f"[User]: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"[Assistant]: {content}")
        prompt = "\n".join(prompt_parts)

        # Generate n responses
        responses = []
        for _ in range(n):
            try:
                response = self.model_instance.generate_content(prompt, generation_config={"temperature": temperature})
                responses.append(type('Choice', (), {'message': type('Message', (), {'content': response.text})})())
            except Exception as e:
                logger.exception(f"Gemini generation failed: {e}")
                raise e

        return responses
