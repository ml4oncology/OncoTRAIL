import os
import pandas as pd
import torch
import logging
from llama_cpp import Llama
from oncotrail.prompt.base_local_runner import BaseLocalLLMRunner
from oncotrail.prompt.base_runner import FEW_SHOT_PHRASE

logger = logging.getLogger(__name__)

# Empty cuda cache
torch.cuda.empty_cache()

class LocalLLMRunnerSequential(BaseLocalLLMRunner):
    """GPU-required runner for local llama-cpp inference."""

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

    # =========================
    # NEW HELPER FUNCTIONS
    # =========================

    def _count_tokens_messages(self, messages):
        """Count tokens for full messages (system + user)."""
        full_text = ""
        for m in messages:
            full_text += m["content"] + "\n"
        return len(self.tokenizer.encode(full_text))


    def _truncate_note_to_fit(self, system_text, note, max_input_tokens):
        """Truncate note from the end to fit token budget."""
        note_tokens = self.tokenizer.encode(note)

        truncated_flag = "\n[NOTE TRUNCATED]"
        truncated_flag_tokens = self.tokenizer.encode(truncated_flag)

        allowed_note_tokens = max_input_tokens - len(self.tokenizer.encode(system_text))

        if allowed_note_tokens <= 0:
            return truncated_flag  # extreme case

        if len(note_tokens) > allowed_note_tokens:
            note_tokens = note_tokens[:allowed_note_tokens - len(truncated_flag_tokens)]
            note = self.tokenizer.decode(note_tokens) + truncated_flag

        return note


    def _remove_last_few_shot_example(self, system_text):
        """Remove last few-shot example (based on 'Outcome:' marker)."""
        split_token = "Clinical Note Summary:"
        parts = system_text.split(split_token)

        if len(parts) <= 1:
            return system_text, False  # nothing to remove

        # Remove last example
        system_text = split_token.join(parts[:-1])

        return system_text, True


    def _adjust_messages_to_fit(self, messages):
        """
        Core logic:
        - enforce token budget
        - remove few-shot examples if needed
        - truncate note if needed
        """
        buffer = 100
        max_input_tokens = self.n_ctx_length - self.max_tokens - buffer

        system_text = messages[0]["content"]
        note = messages[1]["content"]

        # Detect few-shot
        has_few_shot = "Outcome:" in system_text

        # remove few-shot examples if needed
        if has_few_shot:
            while True:
                current_messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": note},
                ]
                n_tokens = self._count_tokens_messages(current_messages)

                if n_tokens <= max_input_tokens:
                    break

                system_text, removed = self._remove_last_few_shot_example(system_text)

                if not removed:
                    break  # no more examples

            # Remove few_shot_phrase if no examples left
            if "Outcome:" not in system_text:
                system_text = system_text.replace(FEW_SHOT_PHRASE, "")

        # truncate note if still too long
        current_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": note},
        ]
        n_tokens = self._count_tokens_messages(current_messages)

        if n_tokens > max_input_tokens:
            note = self._truncate_note_to_fit(system_text, note, max_input_tokens)

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": note},
        ]

    def process_note_for_target(self, note, mrn, treatment_date, target_name, idx, target_val):
        target_name_nospace = target_name.replace("_", "-")
        out_path = f"{self.config.save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{self.config.LLM_name}_prompt{self.config.prompt_num}.csv"
        if os.path.isfile(out_path):
            return

        system_instructions, n_examples_added = self.prepare_system_instructions(target_name, treatment_date, note, mrn)
        logger.info(f"system instructions: {system_instructions}\n")
        logger.info(f"system instructions: {note}\n")
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note},
        ]

        self.generate_and_save_locally(messages, mrn, treatment_date, target_name, target_val, n_examples_added, out_path)
    
    def generate_and_save_locally(self, messages, mrn, treatment_date, target_name, target_val, n_added, out_path):
        llm = self._create_llama()
        results = []

        # adjust messages to fit token budget (handles both few-shot and note truncation)
        messages = self._adjust_messages_to_fit(messages)

        for count in range(self.config.num_samples):
            logger.info(f"Sample count: {count}")

            parsed = None

            # retry up to 10 times
            for attempt in range(10):
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

                # check for valid output
                reason = parsed.get("Reason", None)
                prob = parsed.get("Probability", parsed.get("Prediction", None))

                if (reason is not None and not pd.isna(reason)) and \
                (prob is not None and not pd.isna(prob)):
                    break  # success

                logger.info(f"Retrying due to invalid output (attempt {attempt+1})")

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