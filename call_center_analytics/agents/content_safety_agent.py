"""
Content Safety Agent - Checks transcribed text for inappropriate content.
"""
from typing import Dict, Any
from utils.guardrails import GuardrailsManager
from utils.guardrails_config import GuardrailsConfig


class ContentSafetyAgent:
    """
    Agent responsible for checking transcribed text for inappropriate content
    using OpenAI's Moderation API.
    """
    
    def __init__(self):
        self.name = "Content Safety Agent"
        self.guardrails = GuardrailsManager(GuardrailsConfig.to_dict())
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check transcript for inappropriate content.
        
        Args:
            state: Current agent state containing transcript
            
        Returns:
            Updated state with content safety results
        """
        try:
            transcript = state.get("transcript", "")
            
            if not transcript:
                # No transcript to check
                return {
                    **state,
                    "content_safety_passed": True,
                    "content_safety_details": {},
                    "processing_steps": ["Content Safety: No transcript to check"]
                }
            
            # Run content safety check on the transcript
            results = self.guardrails.check_transcript_safety(transcript)
            
            if not results["passed"]:
                # Content flagged as inappropriate - flag for manual review
                flagged_categories = results.get("flagged_categories", [])
                
                return {
                    **state,
                    "content_safety_passed": False,
                    "content_safety_details": results,
                    "needs_manual_review": True,  # Flag for manual review
                    # NOTE: Don't set "error" field - this would route to END instead of store_flagged
                    "processing_steps": [
                        f"🚨 Content Safety: FLAGGED - {', '.join(flagged_categories)}",
                    ]
                }
            
            # Content is safe
            return {
                **state,
                "content_safety_passed": True,
                "content_safety_details": results,
                "processing_steps": ["Content Safety: Transcript approved"]
            }
            
        except Exception as e:
            # If check fails, log error but allow processing to continue
            return {
                **state,
                "content_safety_passed": True,  # Don't block on error
                "content_safety_details": {"error": str(e)},
                "processing_steps": [f"Content Safety: Check failed - {str(e)}"]
            }
