"""
Summarization Agent - Generates summaries and key points using GPT.
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.config import Config
from utils.models import CallSummary, CallData


class SummarizationAgent:
    """
    Agent responsible for generating summaries and extracting key points
    from call transcriptions using GPT.
    """
    
    def __init__(self):
        self.name = "Summarization Agent"
        self.llm = ChatOpenAI(
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.output_parser = PydanticOutputParser(pydantic_object=CallSummary)
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process call data and generate summary.
        
        Args:
            state: Current agent state containing call_data
            
        Returns:
            Updated state with summary
        """
        try:
            call_data = state.get("call_data")
            
            if not call_data:
                state["error"] = "No call data available for summarization"
                return state
            
            # Generate summary
            summary = self._generate_summary(call_data)
            
            # Update state - return new list for operator.add
            state["summary"] = summary
            
            return {
                **state,
                "processing_steps": ["Summarization: Call summary generated"]
            }
            
        except Exception as e:
            state["error"] = f"Summarization Agent error: {str(e)}"
            return state
    
    def _generate_summary(self, call_data: CallData) -> CallSummary:
        """
        Generate a comprehensive summary of the call.
        
        Args:
            call_data: CallData object containing conversation
            
        Returns:
            CallSummary object
        """
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert call center analyst. Analyze the following call transcript and provide a comprehensive summary.

Call Metadata:
- Call ID: {call_id}
- Caller: {caller_name}
- Agent: {agent_name}
- Duration: {duration}
- Date/Time: {date_time}

Conversation:
{conversation}

Please provide:
1. A brief summary (2-3 sentences) of the overall call
2. Key points discussed in the conversation
3. The main customer issue or request
4. How the issue was resolved (if applicable)
5. Any action items or follow-ups needed

{format_instructions}
"""
        )
        
        # Format the prompt
        format_instructions = self.output_parser.get_format_instructions()
        
        messages = prompt_template.format_messages(
            call_id=call_data.metadata.call_id or "N/A",
            caller_name=call_data.metadata.caller_name or "Unknown",
            agent_name=call_data.metadata.agent_name or "Unknown",
            duration=call_data.metadata.call_duration or "N/A",
            date_time=call_data.metadata.date_time or "N/A",
            conversation=call_data.conversation,
            format_instructions=format_instructions
        )
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            summary = self.output_parser.parse(response.content)
        except Exception as parse_error:
            # Fallback: create summary from raw text
            summary = self._create_fallback_summary(response.content, call_data)
        
        return summary
    
    def _create_fallback_summary(self, raw_text: str, call_data: CallData) -> CallSummary:
        """
        Create a basic summary when structured parsing fails.
        
        Args:
            raw_text: Raw LLM response text
            call_data: Original call data
            
        Returns:
            CallSummary object
        """
        # Extract sections from raw text
        lines = raw_text.split('\n')
        
        brief_summary = ""
        key_points = []
        customer_issue = None
        resolution = None
        action_items = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "brief summary" in line.lower() or "summary:" in line.lower():
                current_section = "summary"
                continue
            elif "key points" in line.lower():
                current_section = "key_points"
                continue
            elif "customer issue" in line.lower() or "main issue" in line.lower():
                current_section = "issue"
                continue
            elif "resolution" in line.lower() or "resolved" in line.lower():
                current_section = "resolution"
                continue
            elif "action items" in line.lower() or "follow-up" in line.lower():
                current_section = "action_items"
                continue
            
            # Add content to appropriate section
            if current_section == "summary" and not brief_summary:
                brief_summary = line
            elif current_section == "key_points" and line.startswith(('-', '•', '*', str)):
                key_points.append(line.lstrip('-•* 0123456789.'))
            elif current_section == "issue" and not customer_issue:
                customer_issue = line
            elif current_section == "resolution" and not resolution:
                resolution = line
            elif current_section == "action_items" and line.startswith(('-', '•', '*', str)):
                action_items.append(line.lstrip('-•* 0123456789.'))
        
        # Fallback values if parsing failed
        if not brief_summary:
            brief_summary = f"Call between {call_data.metadata.caller_name or 'customer'} and {call_data.metadata.agent_name or 'agent'}."
        
        if not key_points:
            key_points = ["Call summary unavailable - structured data could not be extracted"]
        
        return CallSummary(
            brief_summary=brief_summary,
            key_points=key_points,
            customer_issue=customer_issue,
            resolution=resolution,
            action_items=action_items
        )
