"""
Content Safety Guardrails Configuration
"""


class GuardrailsConfig:
    """
    Configuration for Content Safety guardrails.
    """
    
    # Content Safety - Check transcribed text for inappropriate content
    ENABLE_CONTENT_SAFETY = True
    
    # Strict mode uses lower thresholds for flagging content
    CONTENT_SAFETY_STRICT_MODE = False
    
    @classmethod
    def to_dict(cls) -> dict:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary of configuration settings
        """
        return {
            "content_safety": cls.ENABLE_CONTENT_SAFETY,
        }
    
    @classmethod
    def get_summary(cls) -> str:
        """
        Get a human-readable summary of active guardrails.
        
        Returns:
            Summary string
        """
        if cls.ENABLE_CONTENT_SAFETY:
            mode = " (Strict Mode)" if cls.CONTENT_SAFETY_STRICT_MODE else ""
            return f"Active Guardrails: Content Safety{mode}"
        else:
            return "No guardrails active"
