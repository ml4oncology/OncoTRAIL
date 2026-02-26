import os
import pandas as pd
import logging
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from oncotrail.prompt.base_runner import BaseLLMRunner
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ResponseProba(BaseModel):
    Reason: str
    Probability: float

class ResponsePrediction(BaseModel):
    Reason: str
    Prediction: int

class VLLMRunnerOffline(BaseLLMRunner):
    """CPU-compatible runner for vLLM inference."""
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # vLLM-specific state
        self.messages_list = []
        self.mrn_list = []
        self.treatment_date_list = []
        self.target_name_list = []
        self.target_val_list = []
        self.n_examples_added_list = []

    def process_note_for_target(self, note, mrn, treatment_date, target_name, idx, target_val):
        target_name_nospace = target_name.replace("_", "-")
        out_path = f"{self.config.save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{self.config.LLM_name}_prompt{self.config.prompt_num}.csv"
        if os.path.isfile(out_path):
            return

        system_instructions, n_examples_added = self.prepare_system_instructions(target_name, treatment_date, note, mrn)
        logger.info(f"system instructions: {system_instructions}\n")

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note},
        ]

        self.prepare_vllm_batch(messages, mrn, treatment_date, target_name, target_val, n_examples_added)

    def prepare_vllm_batch(self, messages, mrn, treatment_date, target_name, target_val, n_added):
        self.messages_list.append(messages)
        self.mrn_list.append(mrn)
        self.treatment_date_list.append(treatment_date)
        self.target_name_list.append(target_name)
        self.target_val_list.append(target_val)
        self.n_examples_added_list.append(n_added)

    def run(self):
        """Override to add vLLM batch processing after collecting all messages."""
        super().run()  # This will populate the batch lists
        if self.messages_list:  # Only run if we have messages to process
            self.batch_generate_with_vllm()

    def batch_generate_with_vllm(self):    
        """Generate responses using vLLM with structured output only."""
        logger.info(f"Running vllm with {len(self.messages_list)} messages\n")
        
        # Determine response model based on return_val
        if self.return_val == "proba":
            response_model = ResponseProba
        elif self.return_val == "prediction":
            response_model = ResponsePrediction
        else:
            raise ValueError(f"Unsupported return_val: {self.return_val}. Must be 'proba' or 'prediction' for structured output.")
        
        llm = LLM(model=self.config.LLM_path, tokenizer=self.config.tokenizer_path)

        json_schema = response_model.model_json_schema()

        # Create repeated messages for num_samples
        repeated_messages = []
        for messages in self.messages_list:
            for _ in range(self.config.num_samples):
                repeated_messages.append(messages)

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

        for idx, mrn in enumerate(self.mrn_list):
            results = []
            out_samples = outputs[:self.config.num_samples]
            outputs = outputs[self.config.num_samples:]
            for raw in out_samples:
                # Assuming self.utils.process_llm_output can handle the error_response_content
                result = self.utils.process_llm_output(raw, self.return_val)
                result[self.target_name_list[idx]] = self.target_val_list[idx]
                results.append(result)
            df = pd.DataFrame(results)
            df['mrn'] = mrn
            df['treatment_date'] = self.treatment_date_list[idx]
            if self.config.n_few_shot > 0:
                df['n_few_shot_added'] = self.n_examples_added_list[idx]

            out_path = f"{self.config.save_dir}/mrn{mrn}_trtdate{self.treatment_date_list[idx][:10]}_{self.target_name_list[idx].replace('_', '-')}_{self.config.LLM_name}_prompt{self.config.prompt_num}.csv"
            df.to_csv(out_path)