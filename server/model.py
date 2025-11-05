import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class EmolLamaModel:
    """Thin wrapper to load a base model and optionally apply a LoRA adapter.

    Usage:
      loader = EmolLamaModel(base_model="hf/tiiuae/falcon-7b", adapter_path="/path/to/adapter")
      loader.load()
      out = loader.generate("prompt")
    """

    def __init__(self, base_model: str, adapter_path: Optional[str] = None, device: Optional[str] = None):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"Loading tokenizer and base model: {self.base_model} on {self.device}")
        # load base model in 8bit or full precision depending on device
        try:
            # prefer bf16 if available
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = LLM(
            model=self.base_model,
            tokenizer=self.base_model,
            enable_lora=bool(self.adapter_path),
            dtype=dtype,
            gpu_memory_utilization=0.5
        )
        except Exception:
            dtype = torch.float16
            self.model = LLM(
            model=self.base_model,
            tokenizer=self.base_model,
            enable_lora=bool(self.adapter_path),
            dtype=dtype,
            gpu_memory_utilization=0.8
        )


    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
        if self.model is None:
            raise RuntimeError("Model not loaded yet. Call load() first.")
        '''
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Reduced from 2048 for speed
        )
        '''
        sampling_params = SamplingParams(
            temperature = temperature,
            top_p = 0.9,
            top_k = 50,
            repetition_penalty = 1.1,
            max_tokens = max_tokens,
            stop=["\nUser:", "\nuser:", "\n\nUser:", "\nAssistant:", " User:"]
        )
        
        #input_length = inputs['input_ids'].shape[1]
        #inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            prompts = prompt,
            sampling_params = sampling_params,
            lora_request = LoRARequest(lora_name="emollama-mental-health", 
        lora_int_id=1,
        lora_local_path=self.adapter_path) if self.adapter_path else None
        )
        '''
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=10,
            temperature=temperature,
            do_sample=True,
            num_beams=2,          # Reduced from 5
            length_penalty=1.0,   # Neutral length penalty
            early_stopping=True,  # Stop when possible
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )'''
    
        generated_text = outputs[0].outputs[0].text
        # Find last complete sentence
        if '.' in generated_text:
            # Split by period
            sentences = generated_text.split('.')
            
            # Keep all complete sentences
            # (last element after split is either empty or incomplete)
            complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
            
            if complete_sentences:
                generated_text = '. '.join(complete_sentences) + '.'
        return generated_text.strip()
