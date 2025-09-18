# run.py
import uvicorn
from main import MindCircleApp  # Make sure this import path is correct
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Create app instance
    app_instance = MindCircleApp()
    
    # Run server
    uvicorn.run(
        "main:app",  # Changed this line - using string import
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )