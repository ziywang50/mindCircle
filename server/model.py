import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=False)
        # load base model in 8bit or full precision depending on device
        if self.device.startswith("cuda"):
            try:
                # prefer bf16 if available
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                torch_dtype=dtype,
                load_in_8bit=False,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map={"": "cpu"})

        # if adapter specified, try to load as PEFT
        if self.adapter_path:
            print(f"Applying PEFT adapter from {self.adapter_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            except Exception as e:
                print(f"Failed to load PEFT adapter: {e}")

        # ensure model is in eval
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded yet. Call load() first.")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Reduced from 2048 for speed
        )
        
        input_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
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
            )
        
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return generated_text.strip()
