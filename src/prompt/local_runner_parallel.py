import os
import pandas as pd
import torch
import logging
from llama_cpp import Llama
from concurrent.futures import ProcessPoolExecutor, as_completed
from oncotrail.prompt.base_local_runner import BaseLocalLLMRunner

logger = logging.getLogger(__name__)

# Empty cuda cache
torch.cuda.empty_cache()

def _worker_process_mrn(args):
    """Worker function for processing an MRN in a separate process.
    
    Each process creates its own model instance, avoiding CUDA memory conflicts.
    """
    import gc
    import torch
    
    (model_path, chat_format, n_ctx_length, use_flash_attn, max_tokens,
     return_val, response_format_fn, llm_params, messages, num_samples,
     mrn, mrn_idx) = args
    
    llm = None
    outputs = []
    
    try:
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create model in this process
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=0,
            chat_format=chat_format,
            seed=42,  # Fixed seed for reproducibility
            n_ctx=n_ctx_length,
            flash_attn=use_flash_attn,
            n_batch=512,
            use_mmap=True,
            use_mlock=False,
            verbose=False  # Reduce logging noise
        )
        
        for sample_num in range(num_samples):
            try:
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_format_fn,
                    **llm_params
                )
                raw = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Generation error for MRN {mrn}, sample {sample_num}: {e}", flush=True)
                raw = None
            outputs.append(raw)
            
    except Exception as e:
        print(f"CRITICAL: Worker process error for MRN {mrn}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return empty outputs on complete failure
        outputs = [None] * num_samples
        
    finally:
        # Clean up model aggressively
        if llm is not None:
            try:
                llm._sampler.close()
            except:
                pass
            try:
                llm.close()
            except:
                pass
            del llm
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return mrn_idx, outputs

class LocalLLMRunnerParallel(BaseLocalLLMRunner):
    """GPU-required runner for local llama-cpp inference with parallel processing."""
    
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        
        # Batch collection lists (similar to vLLM)
        self.messages_list = []
        self.mrn_list = []
        self.treatment_date_list = []
        self.target_name_list = []
        self.target_val_list = []
        self.n_examples_added_list = []
        self.out_path_list = []
        
        # Parallel processing settings
        self.num_workers = 2  # Reduced for 14B model - adjust based on GPU memory
        
        # Set multiprocessing start method (important for CUDA)
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

    def _create_llama(self, seed=42, split_mode=1):
        """Create a Llama model instance.
        
        Args:
            seed: Random seed
            split_mode: GPU memory split mode
                0 = no split (single GPU)
                1 = split by rows (recommended for multiple instances)
                2 = split by layers
        """
        return Llama(
            model_path=self.config.LLM_path,
            n_gpu_layers=-1,
            main_gpu=0,
            chat_format=self.chat_format,
            seed=seed,
            n_ctx=self.n_ctx_length,
            flash_attn=self.use_flash_attn,
            n_batch=512,
            use_mmap=True,
            use_mlock=False,
            split_mode=split_mode,  # Enable memory splitting
            tensor_split=None  # Let llama.cpp handle it automatically
        )

    def _initialize_models(self):
        """Initialize multiple model instances for parallel processing."""
        logger.info(f"Initializing {self.num_workers} model instances...")
        self.models = []
        for i in range(self.num_workers):
            model = self._create_llama(seed=42 + i)
            self.models.append(model)
        logger.info("Model initialization complete.")

    def _cleanup_models(self):
        """Clean up all model instances."""
        for llm in self.models:
            try:
                llm._sampler.close()
                llm.close()
            except:
                pass
        self.models = []

    def process_note_for_target(self, note, mrn, treatment_date, target_name, idx, target_val):
        target_name_nospace = target_name.replace("_", "-")
        out_path = f"{self.config.save_dir}/mrn{mrn}_trtdate{treatment_date[:10]}_{target_name_nospace}_{self.config.LLM_name}_prompt{self.config.prompt_num}.csv"
        if os.path.isfile(out_path):
            return

        system_instructions, n_examples_added = self.prepare_system_instructions(target_name, treatment_date, note, mrn)
        logger.info(f"system instructions: {system_instructions}\n")
        logger.info(f"note content: {note}\n")
        
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": note},
        ]

        self.prepare_batch(messages, mrn, treatment_date, target_name, target_val, n_examples_added, out_path)

    def prepare_batch(self, messages, mrn, treatment_date, target_name, target_val, n_added, out_path):
        """Collect messages for batch processing (similar to vLLM's prepare_vllm_batch)."""
        self.messages_list.append(messages)
        self.mrn_list.append(mrn)
        self.treatment_date_list.append(treatment_date)
        self.target_name_list.append(target_name)
        self.target_val_list.append(target_val)
        self.n_examples_added_list.append(n_added)
        self.out_path_list.append(out_path)

    def run(self):
        """Override to add batch processing after collecting all messages."""
        super().run()  # This will populate the batch lists
        if self.messages_list:  # Only run if we have messages to process
            self.batch_generate_with_llama_cpp()

    def _process_mrn_samples(self, model_idx, mrn_idx, messages, num_samples):
        """Process all samples for a single MRN on a specific model instance.
        
        This ensures reproducibility by:
        1. Processing all samples for an MRN sequentially on the same model
        2. Resetting the model's seed before processing each MRN
        """
        # Reset the model's random state for this MRN
        self.models[model_idx].set_seed(42)
        
        outputs = []
        for sample_num in range(num_samples):
            try:
                response = self.models[model_idx].create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    response_format=self.config.response_format(self.return_val),
                    **self.config.llm_params
                )
                raw = response['choices'][0]['message']['content']
            except Exception as e:
                logger.warning(f"Generation error for MRN {self.mrn_list[mrn_idx]}, sample {sample_num}, model {model_idx}: {e}")
                raw = None
            outputs.append(raw)
        
        return outputs

    def batch_generate_with_llama_cpp(self):
        """Generate responses using llama-cpp with parallel processes.
        
        Uses ProcessPoolExecutor instead of ThreadPoolExecutor to avoid
        CUDA memory conflicts. Each process creates its own model instance.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Running llama-cpp with {len(self.messages_list)} MRNs")
        logger.info(f"Total samples to generate: {len(self.messages_list) * self.config.num_samples}")
        logger.info(f"Using {self.num_workers} parallel processes")
        
        # Check GPU memory
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            logger.info(f"GPU Memory: {free_mem / 1e9:.2f}GB free / {total_mem / 1e9:.2f}GB total")
            
            # Warn if memory might be tight (14B model needs ~8GB per instance)
            estimated_per_worker = 8e9  # 8GB estimate per 14B model
            if free_mem < estimated_per_worker * self.num_workers:
                logger.warning(f"GPU memory may be insufficient for {self.num_workers} workers!")
                logger.warning(f"Consider reducing num_workers to {int(free_mem / estimated_per_worker)}")
        
        # Prepare arguments for each MRN
        args_list = []
        for idx, messages in enumerate(self.messages_list):
            args = (
                self.config.LLM_path,
                self.chat_format,
                self.n_ctx_length,
                self.use_flash_attn,
                self.max_tokens,
                self.return_val,
                self.config.response_format(self.return_val),
                self.config.llm_params,
                messages,
                self.config.num_samples,
                self.mrn_list[idx],
                idx
            )
            args_list.append(args)
        
        # Process MRNs in parallel using separate processes
        all_mrn_outputs = [None] * len(self.messages_list)
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_mrn_idx = {
                    executor.submit(_worker_process_mrn, args): args[-1]
                    for args in args_list
                }
                
                completed = 0
                for future in as_completed(future_to_mrn_idx):
                    try:
                        mrn_idx, outputs = future.result(timeout=3600)  # 1 hour timeout per MRN
                        all_mrn_outputs[mrn_idx] = outputs
                        completed += 1
                        logger.info(f"Completed MRN {completed}/{len(self.messages_list)}: {self.mrn_list[mrn_idx]}")
                    except Exception as e:
                        mrn_idx = future_to_mrn_idx[future]
                        logger.error(f"Failed to process MRN {self.mrn_list[mrn_idx]}: {e}")
                        # Create empty results for failed MRN
                        all_mrn_outputs[mrn_idx] = [None] * self.config.num_samples
        except Exception as e:
            logger.error(f"ProcessPoolExecutor error: {e}")
            raise
        
        # Save results for each MRN
        for idx in range(len(self.messages_list)):
            if all_mrn_outputs[idx] is None:
                logger.warning(f"Skipping save for MRN {self.mrn_list[idx]} - no outputs generated")
                continue
                
            results = []
            message_outputs = all_mrn_outputs[idx]
            
            for raw in message_outputs:
                parsed = self.utils.process_llm_output(raw, self.return_val)
                parsed[self.target_name_list[idx]] = self.target_val_list[idx]
                results.append(parsed)
            
            # Save results
            df = pd.DataFrame(results)
            df["mrn"] = self.mrn_list[idx]
            df["treatment_date"] = self.treatment_date_list[idx]
            if self.config.n_few_shot > 0:
                df["n_few_shot_added"] = self.n_examples_added_list[idx]
            
            df.to_csv(self.out_path_list[idx])
            logger.info(f"Saved results for MRN {self.mrn_list[idx]} to {self.out_path_list[idx]}")
        
        elapsed = time.time() - start_time
        total_samples = len(self.messages_list) * self.config.num_samples
        logger.info(f"Batch processing complete: {total_samples} samples in {elapsed:.2f}s ({total_samples/elapsed:.2f} samples/sec)")