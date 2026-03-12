# base_local_runner.py
import os
import torch
import logging
from oncotrail.prompt.base_runner import BaseLLMRunner

logger = logging.getLogger(__name__)

torch.cuda.empty_cache()

class BaseLocalLLMRunner(BaseLLMRunner):
    """Shared base for local llama-cpp runners."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.chat_format = self._get_chat_format()
        self.use_flash_attn = torch.cuda.mem_get_info()[1] / 1e5 > 120000

    def _get_chat_format(self):
        name = self.config.LLM_name
        if "Gemma" in name:
            return "gemma"
        elif "Qwen" in name or "QwQ" in name:
            return "chatml"
        elif "Llama" in name:
            return "llama-3"
        else:
            return "llama-2"
