import os
import pandas as pd
import logging
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from oncotrail.prompt.base_vllm_runner import BaseVLLMRunner

logger = logging.getLogger(__name__)

class VLLMRunnerOffline(BaseVLLMRunner):
    """CPU-compatible runner for vLLM inference."""

    def batch_generate_with_vllm(self):    
        """Generate responses using vLLM with structured output only."""
        logger.info(f"Running vllm with {len(self.messages_list)} messages\n")
        response_model = self._get_response_model()
        llm = LLM(model=self.config.LLM_path, tokenizer=self.config.tokenizer_path)
        repeated_messages = self._get_repeated_messages()
        json_schema = response_model.model_json_schema()

        guided = GuidedDecodingParams(json_schema=json_schema)

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            guided_decoding=guided,
            **self.config.llm_params,
            **self.config.extra_body
        )

        logger.info(f"Sending {len(repeated_messages)} requests to vLLM...")

        batch_response = llm.chat(repeated_messages, sampling_params)
        
        # Process structured responses
        outputs = [
            response.outputs[0].text if response is not None else None
            for response in batch_response
        ]
        self._save_batch_results(outputs)