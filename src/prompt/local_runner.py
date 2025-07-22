import os
import pandas as pd
import torch
import logging
from llama_cpp import Llama
from llm_notes_classification.prompt.base_runner import BaseLLMRunner

logger = logging.getLogger(__name__)

# Empty cuda cache
torch.cuda.empty_cache()


class LocalLLMRunner(BaseLLMRunner):
    """GPU-required runner for local llama-cpp inference."""
    
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

    def _create_llama(self):
        return Llama(
            model_path=self.config.LLM_path,
            n_gpu_layers=-1,
            main_gpu=0,
            chat_format=self.chat_format,
            seed=42,
            n_ctx=self.n_ctx_length,
            flash_attn=self.use_flash_attn
        )

    def process_note_for_target(self, note, mrn, treatment_date, target_name, idx, target_val):
        target_name_nospace = target_name.replace("_", "-")
        out_path = f"{self.config.save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{self.config.LLM_name}_prompt{self.config.prompt_num}.csv"
        if os.path.isfile(out_path):
            return

        system_instructions, n_examples_added = self.prepare_system_instructions(target_name, treatment_date, note)
        logger.info(f"system instructions: {system_instructions}\n")

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note},
        ]

        self.generate_and_save_locally(messages, mrn, treatment_date, target_name, target_val, n_examples_added, out_path)

    def generate_and_save_locally(self, messages, mrn, treatment_date, target_name, target_val, n_added, out_path):
        llm = self._create_llama()
        results = []
        for count in range(self.config.num_samples):
            logger.info(f"Sample count: {count}")
            try:
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    response_format=self.config.response_format(self.return_val),
                    **self.config.llm_params
                )
                raw = response['choices'][0]['message']['content']
            except Exception as e:
                logger.warning(f"Generation error: {e}")
                raw = None
            parsed = self.utils.process_llm_output(raw, self.return_val)
            parsed[target_name] = target_val
            results.append(parsed)

        df = pd.DataFrame(results)
        df["mrn"] = mrn
        df["treatment_date"] = treatment_date
        if self.config.n_few_shot > 0:
            df["n_few_shot_added"] = n_added
        df.to_csv(out_path)

        try:
            llm._sampler.close()
            llm.close()
        except:
            llm = None