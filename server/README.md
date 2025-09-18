# mindCircle FastAPI EmolLama Server

This folder contains a minimal FastAPI server to serve the EmolLama model.

Files
- `main.py` - FastAPI application with `/generate` and `/health` endpoints.
- `model.py` - Model loader that loads a base model and optionally applies a PEFT adapter.
- `requirements.txt` - Python dependencies.
- `start.sh` - Simple script to run the server with uvicorn.

Usage
1. Install dependencies (prefer a virtualenv):

```bash
pip install -r mindCircle/server/requirements.txt
```

2. Set environment variables as needed:
- `BASE_MODEL` - HuggingFace model ID or local path (defaults to `emollama_finetune/emollama-mental-health-lora_latest`)
- `ADAPTER_PATH` - optional PEFT adapter directory
- `DEVICE` - `cuda` or `cpu`

3. Start server:

```bash
bash mindCircle/server/start.sh
```

Endpoints
- POST `/generate` {prompt, max_tokens, temperature}
- GET `/health`

Notes
- This is a minimal scaffold. For production, add batching, concurrency controls, authentication, and better device/precision management.
