import os
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional, List, Any
from functools import wraps
from model import EmolLamaModel
from transformers import pipeline

class GenRequest(BaseModel):
    user_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class ContextManager:
    def __init__(self, max_messages: int=3):
        self.contexts: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_messages)
        )
    
    def add_exchange(self, user_id:str, prompt:str, response:str):
        self.contexts[user_id].append((prompt, response))

    def get_recent(self, user_id:str)-> str:
        if not self.contexts[user_id]:
            return "No past memories"
        context = []
        for prompt, response in self.contexts[user_id]:
            context.extend([
                f"User: {prompt}",
                f"Assistant: {response}"
            ])
        return "\n".join(context)

class MindCircleApp:
    @staticmethod
    def log_time(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            print(f"\nTotal {func.__name__} execution time: {end-start:.2f} seconds")
            return result
        return wrapper

    def __init__(self):
        self.app = FastAPI(title="mindCircle EmolLama API")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )
        
        self.model: Optional[EmolLamaModel] = None
        self.context_manager: Optional[ContextManager] = None
        
        # Register routes with timing wrapper
        self.app.post("/generate")(self.log_time(self.generate))
        self.app.get("/health")(self.health)
        self.app.on_event("startup")(self.startup_event)

    def extract_important_detail(self, response: str, max_len: int = 100) -> str:
        """Extract first meaningful sentence as key information."""
        first_sentence = response.split('.')[0].strip()
        return (first_sentence[:max_len-3] + "...") if len(first_sentence) > max_len else first_sentence

    async def startup_event(self):
        """Initialize all components on startup."""
        self.context_manager = ContextManager(max_messages=5)
        
        # Initialize model with logging
        base = os.environ.get("BASE_MODEL", "lzw1008/Emollama-7b")
        default_adapter = "/root/autodl-tmp/mindCircle/emollama-mental-health-lora_latest"
        adapter_path = os.environ.get("ADAPTER_PATH", default_adapter)
        
        print(f"Loading base model from: {base}")
        if adapter_path:
            print(f"Will apply adapter from: {adapter_path}")
            if not os.path.exists(adapter_path):
                print(f"WARNING: Adapter path does not exist: {adapter_path}")
        
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or '0') else "cpu"
        print(f"Using device: {device}")
        
        self.model = EmolLamaModel(base_model=base, adapter_path=adapter_path, device=device)
        try:
            self.model.load()
            time.sleep(5)
            if self.model is None:
                raise RuntimeError("Model failed to load.")
            # Verify model loaded
            test_output = self.model.generate("Test: Are you ready to help? Counselor:", max_tokens=10)
            print(f"Model test output: {test_output}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    @log_time  # Add decorator here too
    async def generate(self, req: GenRequest):
        """Handle generation requests with memory context."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:            
            t1 = time.time()
            INSTRUCTION = """You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. 
The assistant gives helpful, comprehensive, and appropriate answers to the user's questions."""

            past = self.context_manager.get_recent(req.user_id)
            if past != "No past memories":
                conversation_history = f"""[Conversation History]: 
{past}

[Current Question]: """
            else:
                conversation_history = ""
            enhanced_prompt = f"""{INSTRUCTION}
{conversation_history}
User: {req.prompt.strip()}
Assistant:"""

            t2 = time.time()
            print(f"Prompt preparation took: {t2-t1:.2f}s")

            # Generate
            t3 = time.time()
            out = self.model.generate(enhanced_prompt, max_tokens=req.max_tokens, temperature=req.temperature)
            self.context_manager.add_exchange(req.user_id, req.prompt.strip(), out)
            t4 = time.time()
            print(f"Model generation took: {t4-t3:.2f}s")

            # Update conversation context
            # No context management
            return {"text": out}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def health(self):
        """Health check endpoint."""
        return {"status": "ok", "loaded": self.model is not None}

mindcircle = MindCircleApp()
app = mindcircle.app  # This is what uvicorn will import
