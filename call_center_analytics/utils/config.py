"""
Configuration settings for the Call Center AI Assistant system.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the application."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
    
    # Temperature settings
    TEMPERATURE = 0.3  # Lower temperature for more consistent outputs
    
    # Supported file formats
    SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    SUPPORTED_TEXT_FORMATS = [".txt"]
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        return True


# Validate configuration on import
Config.validate()
