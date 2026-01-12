"""
Quality Scoring Agent - Evaluates tone, professionalism, and resolution.
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.config import Config
from utils.models import QualityScore, CallData, CallSummary


class QualityScoringAgent:
    """
    Agent responsible for evaluating call quality based on tone, 
    professionalism, and resolution using a structured rubric.
    """
    
    def __init__(self):
        self.name = "Quality Scoring Agent"
        self.llm = ChatOpenAI(
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.output_parser = PydanticOutputParser(pydantic_object=QualityScore)
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process call data and generate quality score.
        
        Args:
            state: Current agent state containing call_data and summary
            
        Returns:
            Updated state with quality_score
        """
        try:
            call_data = state.get("call_data")
            summary = state.get("summary")
            
            if not call_data:
                state["error"] = "No call data available for quality scoring"
                return state
            
            # Generate quality score
            quality_score = self._evaluate_quality(call_data, summary)
            
            # Check if scoring failed (None returned)
            if quality_score is None:
                # Scoring failed - mark for manual review
                return {
                    **state,
                    "quality_score": None,
                    "needs_manual_review": True,
                    "processing_steps": [
                        " Quality Scoring: Failed - Unable to extract scores from LLM response. Call saved for manual review."
                    ]
                }
            
            # Update state - return new list for operator.add
            state["quality_score"] = quality_score
            
            return {
                **state,
                "processing_steps": ["Quality Scoring: Call quality evaluated"]
            }
            
        except Exception as e:
            # Unexpected error - also mark for manual review
            return {
                **state,
                "quality_score": None,
                "needs_manual_review": True,
                "processing_steps": [
                    f"⚠️ Quality Scoring: Error - {str(e)}. Call saved for manual review."
                ]
            }
    
    def _evaluate_quality(self, call_data: CallData, summary: CallSummary = None) -> QualityScore:
        """
        Evaluate the quality of the call based on multiple criteria.
        
        Args:
            call_data: CallData object containing conversation
            summary: Optional CallSummary object
            
        Returns:
            QualityScore object
        """
        # Create comprehensive prompt with scoring rubric
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert call center quality assurance analyst. Evaluate the following call based on a structured rubric.

Call Metadata:
- Agent: {agent_name}
- Caller: {caller_name}
- Duration: {duration}

Conversation:
{conversation}

{summary_section}

SCORING RUBRIC (0-10 scale for each criterion):

1. TONE AND EMPATHY (0-10):
   - 9-10: Consistently warm, empathetic, and understanding throughout
   - 7-8: Generally positive tone with good empathy
   - 5-6: Neutral tone, adequate empathy
   - 3-4: Occasionally cold or dismissive
   - 0-2: Consistently poor tone, lacking empathy

2. PROFESSIONALISM (0-10):
   - 9-10: Excellent communication skills, proper grammar, courteous language
   - 7-8: Professional with minor lapses
   - 5-6: Adequate professionalism
   - 3-4: Several unprofessional moments
   - 0-2: Unprofessional behavior or language

3. PROBLEM RESOLUTION / CALL EFFECTIVENESS (0-10):
   CONTEXT-AWARE SCORING:
   
   For calls WITH a problem/issue:
   - 9-10: Issue fully resolved, customer clearly satisfied
   - 7-8: Issue resolved with minor follow-up needed
   - 5-6: Partial resolution or unclear outcome
   - 3-4: Issue not adequately addressed
   - 0-2: No resolution or made situation worse
   
   For calls WITHOUT a problem (informational, inquiry, status check):
   - 9-10: Question fully answered, information clearly provided, customer satisfied
   - 7-8: Question answered with minor gaps or needed clarification
   - 5-6: Adequate response but could be more complete
   - 3-4: Incomplete or unclear information provided
   - 0-2: Failed to address the inquiry or provided incorrect information
   
   NOTE: If there was NO issue or problem to resolve, focus on how effectively the agent 
   handled the inquiry, provided information, or addressed the customer's needs.

4. RESPONSE APPROPRIATENESS (0-10):
   - 9-10: All responses relevant, clear, and helpful
   - 7-8: Mostly appropriate responses
   - 5-6: Some responses could be improved
   - 3-4: Several inappropriate or unclear responses
   - 0-2: Consistently poor or off-topic responses

Provide:
1. Individual scores for each criterion (0-10)
2. Overall score (average of all criteria, 0-10)
3. Detailed feedback explaining the scores
4. List of strengths demonstrated
5. List of areas for improvement

IMPORTANT: Adapt your scoring to the call type. Not all calls have problems to solve - 
some are informational, transactional, or proactive. Score based on effectiveness.

{format_instructions}
"""
        )
        
        # Build summary section if available
        summary_section = ""
        if summary:
            summary_section = f"""
Call Summary:
{summary.brief_summary}

Key Points:
{chr(10).join(f'- {point}' for point in summary.key_points)}

Customer Issue: {summary.customer_issue or 'N/A'}
Resolution: {summary.resolution or 'N/A'}
"""
        
        # Format the prompt
        format_instructions = self.output_parser.get_format_instructions()
        
        messages = prompt_template.format_messages(
            agent_name=call_data.metadata.agent_name or "Unknown",
            caller_name=call_data.metadata.caller_name or "Unknown",
            duration=call_data.metadata.call_duration or "N/A",
            conversation=call_data.conversation,
            summary_section=summary_section,
            format_instructions=format_instructions
        )
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            quality_score = self.output_parser.parse(response.content)
        except Exception as parse_error:
            # Fallback: create score from raw text
            quality_score = self._create_fallback_score(response.content)
            # If fallback also fails (returns None), return None
            if quality_score is None:
                return None
        
        return quality_score
    
    def _create_fallback_score(self, raw_text: str) -> QualityScore:
        """
        Create a basic quality score when structured parsing fails.
        Only uses scores that can be extracted from text - no arbitrary defaults.
        
        Args:
            raw_text: Raw LLM response text
            
        Returns:
            QualityScore object with extracted values or None indicators for missing scores
        """
        import re
        
        # Track which scores were successfully extracted
        extracted_scores = {}
        
        # Look for score patterns
        tone_match = re.search(r'tone[^:]*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
        if tone_match:
            extracted_scores['tone'] = float(tone_match.group(1))
        
        prof_match = re.search(r'professionalism[^:]*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
        if prof_match:
            extracted_scores['professionalism'] = float(prof_match.group(1))
        
        res_match = re.search(r'resolution[^:]*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
        if res_match:
            extracted_scores['resolution'] = float(res_match.group(1))
        
        resp_match = re.search(r'response[^:]*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
        if resp_match:
            extracted_scores['response_time'] = float(resp_match.group(1))
        
        overall_match = re.search(r'overall[^:]*:\s*(\d+(?:\.\d+)?)', raw_text, re.IGNORECASE)
        if overall_match:
            extracted_scores['overall'] = float(overall_match.group(1))
        
        # Calculate overall as mean of extracted individual scores (if not found directly)
        if 'overall' not in extracted_scores and extracted_scores:
            individual_scores = [
                extracted_scores.get('tone'),
                extracted_scores.get('professionalism'),
                extracted_scores.get('resolution'),
                extracted_scores.get('response_time')
            ]
            # Only include scores that were actually found
            valid_scores = [s for s in individual_scores if s is not None]
            if valid_scores:
                extracted_scores['overall'] = sum(valid_scores) / len(valid_scores)
        
        # Check if we have enough data to create a reliable score
        # We need at least the overall score OR at least 2 individual scores
        has_enough_data = (
            'overall' in extracted_scores or 
            len(extracted_scores) >= 2
        )
        
        if not has_enough_data:
            # Not enough data extracted - return None to indicate scoring should be skipped
            # This will be handled by the workflow
            return None
        
        # Extract strengths and improvements
        strengths = []
        improvements = []
        
        lines = raw_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'strength' in line.lower():
                current_section = 'strengths'
                continue
            elif 'improvement' in line.lower() or 'area' in line.lower():
                current_section = 'improvements'
                continue
            
            if current_section == 'strengths' and line.startswith(('-', '•', '*')):
                strengths.append(line.lstrip('-•* '))
            elif current_section == 'improvements' and line.startswith(('-', '•', '*')):
                improvements.append(line.lstrip('-•* '))
        
        if not strengths:
            strengths = ["Professional communication"]
        if not improvements:
            improvements = ["Manual review needed - automated scoring incomplete"]
        
        # Use extracted scores or 0.0 for missing ones (will be marked as unreliable)
        return QualityScore(
            overall_score=extracted_scores.get('overall', 0.0),
            tone_score=extracted_scores.get('tone', 0.0),
            professionalism_score=extracted_scores.get('professionalism', 0.0),
            resolution_score=extracted_scores.get('resolution', 0.0),
            response_time_score=extracted_scores.get('response_time', 0.0),
            feedback=f"⚠️ Partial extraction ({len(extracted_scores)} scores found). " + 
                     (raw_text[:400] if len(raw_text) > 400 else raw_text),
            strengths=strengths,
            areas_for_improvement=improvements
        )
