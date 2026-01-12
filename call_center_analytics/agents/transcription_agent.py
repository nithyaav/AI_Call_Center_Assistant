"""
Transcription Agent - Converts audio to text using OpenAI Whisper.
"""
import os
from typing import Dict, Any
from openai import OpenAI
from utils.config import Config


class TranscriptionAgent:
    """
    Agent responsible for converting audio files to text using Whisper API.
    """
    
    def __init__(self):
        self.name = "Transcription Agent"
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.WHISPER_MODEL
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio file and transcribe it to text.
        
        Args:
            state: Current agent state containing input_content (file path)
            
        Returns:
            Updated state with transcription
        """
        try:
            audio_file_path = state.get("input_content", "")
            
            if not audio_file_path:
                state["error"] = "No audio file path provided"
                return state
            
            if not os.path.exists(audio_file_path):
                state["error"] = f"Audio file not found: {audio_file_path}"
                return state
            
            # Check file extension
            file_ext = os.path.splitext(audio_file_path)[1].lower()
            if file_ext not in Config.SUPPORTED_AUDIO_FORMATS:
                state["error"] = f"Unsupported audio format: {file_ext}"
                return state
            
            # Transcribe audio
            transcription = self._transcribe_audio(audio_file_path)
            
            # Update state - return new list for operator.add
            state["transcript"] = transcription
            
            return {
                **state,
                "processing_steps": ["Transcription: Audio converted to text"]
            }
            
        except Exception as e:
            state["error"] = f"Transcription Agent error: {str(e)}"
            return state
    
    def _transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file using OpenAI Whisper API.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="text"
                )
            
            return transcript
            
        except Exception as e:
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    def validate_audio_file(self, file_path: str) -> tuple[bool, str]:
        """
        Validate audio file before processing.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in Config.SUPPORTED_AUDIO_FORMATS:
            return False, f"Unsupported format: {file_ext}. Supported: {Config.SUPPORTED_AUDIO_FORMATS}"
        
        # Check file size (Whisper has a 25 MB limit)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 25:
            return False, f"File too large: {file_size_mb:.2f} MB. Maximum: 25 MB"
        
        return True, ""
