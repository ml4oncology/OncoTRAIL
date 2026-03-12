import os
import pandas as pd
import logging
import asyncio
from openai import OpenAI, AsyncOpenAI
from oncotrail.prompt.base_vllm_runner import BaseVLLMRunner

logger = logging.getLogger(__name__)

class VLLMRunnerOnline(BaseVLLMRunner):
    """CPU-compatible runner for vLLM inference."""

    def batch_generate_with_vllm(self):    
        """Generate responses using vLLM with structured output only."""
        logger.info(f"Running vllm with {len(self.messages_list)} messages\n")
        response_model = self._get_response_model()
        logger.info(f"Base URL for OpenAI client: {self.config.base_url}")
        client = OpenAI(base_url=self.config.base_url, api_key="EMPTY", timeout=7200)
        repeated_messages = self._get_repeated_messages()
        
        # Define get_response function with error handling
        def get_response_with_error_handling(chat):
            try:
                # response = client.beta.chat.completions.parse(
                #     model=self.config.vllm_model_name,
                #     messages=chat,
                #     response_format=response_model,
                #     max_tokens=self.max_tokens, # This is crucial: adjust if needed
                #     extra_body=self.config.extra_body,
                #     **self.config.llm_params
                # )

                self.config.extra_body["guided_json"] = response_model.model_json_schema()
                response = client.chat.completions.create(
                    model=self.config.vllm_model_name,
                    messages=chat,
                    max_tokens=self.max_tokens, # This is crucial: adjust if needed
                    extra_body=self.config.extra_body,
                    **self.config.llm_params
                )

                return response
            
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM call: {e}. Returning a default error response.")
                return None
        
        logger.info(f"Sending {len(repeated_messages)} requests to vLLM server...")
        batch_response = [get_response_with_error_handling(chat) for chat in repeated_messages]
        
        # Process structured responses
        outputs = [
            response.choices[0].message.content.strip() if response is not None else None
            for response in batch_response
        ]
        self._save_batch_results(outputs)