import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated model name
    
    # File Upload Settings
    MAX_FILE_SIZE = 200  # MB
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls', 'json', 'parquet']
    UPLOAD_DIR = "data/uploads"
    
    # Analysis Settings
    MAX_ROWS_DISPLAY = 1000
    DEFAULT_SAMPLE_SIZE = 10000
    
    # Visualization Settings
    DEFAULT_CHART_HEIGHT = 500
    DEFAULT_CHART_WIDTH = 800
    
    @staticmethod
    def ensure_upload_dir():
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
