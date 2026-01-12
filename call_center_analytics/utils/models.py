"""
Shared data models for the Call Center AI Assistant system.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class CallMetadata(BaseModel):
    """Metadata extracted from a call."""
    call_id: Optional[str] = Field(None, description="Unique call identifier")
    caller_name: Optional[str] = Field(None, description="Name of the caller")
    agent_name: Optional[str] = Field(None, description="Name of the agent")
    call_duration: Optional[str] = Field(None, description="Duration of the call")
    date_time: Optional[str] = Field(None, description="Date and time of the call")


class ValidationAndMetadata(BaseModel):
    """Combined validation result and metadata extraction."""
    is_valid: bool = Field(..., description="Whether this is a valid call center conversation")
    validation_reason: str = Field(..., description="Explanation of why valid or invalid")
    metadata: Optional[CallMetadata] = Field(None, description="Extracted metadata (only if valid)")
    
    
class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    speaker: str = Field(..., description="Speaker role (Agent/Caller/Customer)")
    text: str = Field(..., description="What was said")


class CallData(BaseModel):
    """Complete call data with metadata and conversation."""
    metadata: CallMetadata
    conversation: str = Field(..., description="Full conversation text")
    conversation_turns: List[ConversationTurn] = Field(default_factory=list, description="Parsed conversation turns")


class CallSummary(BaseModel):
    """Summary of a call."""
    brief_summary: str = Field(..., description="Brief overview of the call")
    key_points: List[str] = Field(..., description="Key points from the conversation")
    customer_issue: Optional[str] = Field(None, description="Main customer issue or request")
    resolution: Optional[str] = Field(None, description="How the issue was resolved")
    action_items: List[str] = Field(default_factory=list, description="Action items or follow-ups")


class QualityScore(BaseModel):
    """Quality scoring for a call."""
    overall_score: float = Field(..., description="Overall quality score (0-10)")
    tone_score: float = Field(..., description="Tone and empathy score (0-10)")
    professionalism_score: float = Field(..., description="Professionalism score (0-10)")
    resolution_score: float = Field(..., description="Problem resolution or call effectiveness score (0-10)")
    response_time_score: float = Field(..., description="Response appropriateness score (0-10)")
    feedback: str = Field(..., description="Detailed feedback on the call quality")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas needing improvement")


class AgentState(BaseModel):
    """State object passed between agents in the LangGraph."""
    # Input
    input_type: str = Field(..., description="Type of input: 'text' or 'audio'")
    input_content: str = Field(..., description="File path or content")
    
    # Intermediate states
    transcription: Optional[str] = None
    call_data: Optional[CallData] = None
    summary: Optional[CallSummary] = None
    quality_score: Optional[QualityScore] = None
    
    # Quality and review flags
    needs_manual_review: Optional[bool] = Field(False, description="Flag indicating call needs manual review")
    
    # Error handling
    error: Optional[str] = None
    
    # Metadata
    processing_steps: List[str] = Field(default_factory=list, description="Steps completed")
    
    class Config:
        arbitrary_types_allowed = True
