"""
Agent Workflow - Orchestrates all agents using LangGraph with Content Safety.
"""
from typing import Dict, Any, TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from agents.call_intake_agent import CallIntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.content_safety_agent import ContentSafetyAgent
from agents.summarization_agent import SummarizationAgent
from agents.quality_scoring_agent import QualityScoringAgent
from agents.data_storage_agent import DataStorageAgent


class GraphState(TypedDict):
    """State definition for the agent graph."""
    # Input
    input_type: str
    input_content: str
    
    # Intermediate states
    transcript: str
    call_data: Any
    summary: Any
    quality_score: Any
    
    # Storage info
    storage_path: str
    
    # Quality and review flags
    needs_manual_review: bool
    
    # Error handling
    error: str
    validation_failed: bool
    
    # Content Safety
    content_safety_passed: bool
    content_safety_details: Dict[str, Any]
    
    # Metadata
    processing_steps: Annotated[Sequence[str], operator.add]


class CallCenterWorkflow:
    """
    Orchestrates the multi-agent workflow using LangGraph.
    Handles routing between agents based on input type (text vs audio).
    """
    
    def __init__(self):
        """Initialize all agents and build the workflow graph."""
        # Initialize agents
        self.transcription_agent = TranscriptionAgent()
        self.content_safety_agent = ContentSafetyAgent()
        self.call_intake_agent = CallIntakeAgent()
        self.summarization_agent = SummarizationAgent()
        self.quality_scoring_agent = QualityScoringAgent()
        self.data_storage_agent = DataStorageAgent()
        
        # Build the graph
        self.workflow = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with Content Safety check after transcription.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes (agents)
        workflow.add_node("transcription", self._transcription_node)
        workflow.add_node("content_safety", self._content_safety_node)
        workflow.add_node("call_intake", self._call_intake_node)
        workflow.add_node("summarization", self._summarization_node)
        workflow.add_node("quality_scoring", self._quality_scoring_node)
        workflow.add_node("data_storage", self._data_storage_node)
        
        # Set entry point with conditional routing
        workflow.set_conditional_entry_point(
            self._route_entry,
            {
                "transcription": "transcription",
                "content_safety": "content_safety"  # Text goes directly to content safety
            }
        )
        
        # Add edges
        # Transcription -> Content Safety (check after transcription)
        workflow.add_edge("transcription", "content_safety")
        
        # Content Safety -> Conditional (check if content is safe)
        workflow.add_conditional_edges(
            "content_safety",
            self._should_continue_after_safety_check,
            {
                "continue": "call_intake",
                "store_flagged": "data_storage",  # Store flagged content for manual review
                "end": END
            }
        )
        
        # Call Intake -> Conditional (check for validation failure)
        workflow.add_conditional_edges(
            "call_intake",
            self._should_continue_after_intake,
            {
                "continue": "summarization",
                "end": END
            }
        )
        
        # Summarization -> Conditional (check if agent name exists for quality scoring)
        workflow.add_conditional_edges(
            "summarization",
            self._should_do_quality_scoring,
            {
                "score": "quality_scoring",
                "skip": "data_storage"  # Still store data even if scoring skipped
            }
        )
        
        # Quality Scoring -> Data Storage
        workflow.add_edge("quality_scoring", "data_storage")
        
        # Data Storage -> END
        workflow.add_edge("data_storage", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _route_entry(self, state: Dict[str, Any]) -> str:
        """
        Route to the appropriate entry point based on input type.
        
        Args:
            state: Current state
            
        Returns:
            Name of the next node
        """
        input_type = state.get("input_type", "")
        
        if input_type == "audio":
            return "transcription"  # Audio needs transcription first
        else:  # text
            return "content_safety"  # Text goes directly to content safety check
    
    def _should_continue_after_safety_check(self, state: Dict[str, Any]) -> str:
        """
        Determine if workflow should continue after content safety check.
        
        Args:
            state: Current state
            
        Returns:
            "continue" to proceed with normal processing, 
            "store_flagged" to store flagged content for manual review,
            "end" to stop completely (on error)
        """
        # If content safety check failed, flag for manual review and store
        # (Check this FIRST before checking for other errors)
        if not state.get("content_safety_passed", True):
            return "store_flagged"
        
        # If there's a system error (not content safety issue), stop processing
        if state.get("error"):
            return "end"
        
        # Otherwise continue to normal call intake
        return "continue"
    
    def _should_continue_after_intake(self, state: Dict[str, Any]) -> str:
        """
        Determine if workflow should continue after call intake validation.
        
        Args:
            state: Current state
            
        Returns:
            "continue" to proceed with processing, "end" to stop
        """
        # If there's an error or validation failed, stop processing
        if state.get("error") or state.get("validation_failed"):
            return "end"
        
        # If call_data is missing, stop processing
        if not state.get("call_data"):
            return "end"
        
        # Otherwise continue to summarization
        return "continue"
    
    def _should_do_quality_scoring(self, state: Dict[str, Any]) -> str:
        """
        Determine if quality scoring should be performed.
        Always performs quality scoring if call_data exists (even without agent name).
        
        Args:
            state: Current state
            
        Returns:
            "score" to proceed with quality scoring, "skip" to skip
        """
        # Check if call_data exists
        call_data = state.get("call_data")
        if not call_data:
            return "skip"
        
        # Always proceed with quality scoring (agent name is optional)
        return "score"
    
    def _transcription_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcription agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        if state.get("error"):
            return state
        
        return self.transcription_agent.process(state)
    
    def _content_safety_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content safety agent to check transcript.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with content safety results
        """
        if state.get("error"):
            return state
        
        return self.content_safety_agent.process(state)
    
    def _call_intake_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute call intake agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        if state.get("error"):
            return state
        
        return self.call_intake_agent.process(state)
    
    def _summarization_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute summarization agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        if state.get("error"):
            return state
        
        return self.summarization_agent.process(state)
    
    def _quality_scoring_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute quality scoring agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        if state.get("error"):
            return state
        
        return self.quality_scoring_agent.process(state)
    
    def _data_storage_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data storage agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        if state.get("error"):
            return state
        
        return self.data_storage_agent.process(state)
    
    def process(self, input_type: str, input_content: str) -> Dict[str, Any]:
        """
        Process input through the entire workflow.
        
        Args:
            input_type: Type of input ('text' or 'audio')
            input_content: Content (file path or text content)
            
        Returns:
            Final state with all results
        """
        # Initialize state
        initial_state = {
            "input_type": input_type,
            "input_content": input_content,
            "transcript": None,
            "call_data": None,
            "summary": None,
            "quality_score": None,
            "storage_path": None,
            "error": None,
            "validation_failed": False,
            "needs_manual_review": False,
            "content_safety_passed": True,
            "content_safety_details": {},
            "processing_steps": []
        }
        
        # For text input, set transcript directly
        if input_type == "text":
            initial_state["transcript"] = input_content
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            initial_state["error"] = f"Workflow error: {str(e)}"
            return initial_state
    
    async def process_async(self, input_type: str, input_content: str) -> Dict[str, Any]:
        """
        Process input through the entire workflow asynchronously.
        
        Args:
            input_type: Type of input ('text' or 'audio')
            input_content: Content (file path or text content)
            
        Returns:
            Final state with all results
        """
        # Initialize state
        initial_state = {
            "input_type": input_type,
            "input_content": input_content,
            "transcript": None,
            "call_data": None,
            "summary": None,
            "quality_score": None,
            "storage_path": None,
            "error": None,
            "validation_failed": False,
            "needs_manual_review": False,
            "content_safety_passed": True,
            "content_safety_details": {},
            "processing_steps": []
        }
        
        # For text input, set transcript directly
        if input_type == "text":
            initial_state["transcript"] = input_content
        
        # Run the workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return final_state
        except Exception as e:
            initial_state["error"] = f"Workflow error: {str(e)}"
            return initial_state
