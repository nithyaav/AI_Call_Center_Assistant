"""
Content Safety Guardrails Module

This module provides Content Safety checking using OpenAI's Moderation API
to detect inappropriate content in transcribed call text.
"""
from typing import Dict, Any, Tuple
from openai import OpenAI
from utils.config import Config


class GuardrailViolation(Exception):
    """Exception raised when a guardrail check fails."""
    
    def __init__(self, message: str, violation_type: str, details: Dict[str, Any] = None):
        self.message = message
        self.violation_type = violation_type
        self.details = details or {}
        super().__init__(self.message)


class ContentSafetyGuardrails:
    """
    Uses OpenAI's moderation API to detect inappropriate content in transcripts.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.flagged_categories = []
    
    def check_content(self, text: str, strict_mode: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Check transcribed text for inappropriate content using OpenAI moderation.
        
        Args:
            text: Transcribed text to check
            strict_mode: If True, flag content with lower thresholds
            
        Returns:
            Tuple of (is_safe, details_dict)
        """
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Get flagged categories
            flagged = []
            for category, is_flagged in result.categories.model_dump().items():
                if is_flagged:
                    flagged.append(category)
            
            # In strict mode, also check category scores against lower thresholds
            if strict_mode and not flagged:
                scores = result.category_scores.model_dump()
                for category, score in scores.items():
                    if score > 0.1:  # Lower threshold for strict mode
                        flagged.append(f"{category}_warning")
            
            is_safe = not bool(flagged)
            
            details = {
                "flagged": result.flagged,
                "categories": flagged,
                "all_scores": result.category_scores.model_dump() if not is_safe else {}
            }
            
            self.flagged_categories = flagged
            return is_safe, details
            
        except Exception as e:
            # If moderation check fails, log error but don't block processing
            print(f"Moderation check failed: {e}")
            return True, {"error": str(e), "check_failed": True}


class GuardrailsManager:
    """
    Manager for Content Safety guardrails.
    Checks transcribed text for inappropriate content.
    """
    
    def __init__(self, config: Dict[str, bool] = None):
        """
        Initialize guardrails manager.
        
        Args:
            config: Configuration dict to enable/disable content safety
        """
        default_config = {
            "content_safety": True,
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize content safety component
        self.content_safety = ContentSafetyGuardrails() if self.config["content_safety"] else None
    
    def check_transcript_safety(self, transcript: str) -> Dict[str, Any]:
        """
        Run content safety check on transcribed text.
        
        Args:
            transcript: Transcribed text to check
            
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "flagged_categories": []
        }
        
        # Content safety check
        if self.config["content_safety"] and self.content_safety:
            is_safe, details = self.content_safety.check_content(transcript)
            if not is_safe:
                results["passed"] = False
                results["flagged_categories"] = details.get("categories", [])
                results["violations"].append({
                    "type": "content_safety",
                    "message": "Transcript contains inappropriate content",
                    "details": details
                })
        
        return results
