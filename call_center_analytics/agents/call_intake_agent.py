"""
Call Intake Agent - Validates input formats and extracts metadata using LLM.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.models import CallMetadata, CallData, ConversationTurn, AgentState, ValidationAndMetadata
from utils.config import Config


class CallIntakeAgent:
    """
    Agent responsible for validating input formats and extracting metadata
    from call transcriptions using LLM for intelligent extraction.
    Uses a single LLM call for both validation and metadata extraction (efficient!).
    """
    
    def __init__(self):
        """Initialize the Call Intake Agent with LLM."""
        self.name = "Call Intake Agent"
        self.llm = ChatOpenAI(
            model=Config.GPT_MODEL,
            temperature=0.1,  # Low temperature for consistent extraction
            openai_api_key=Config.OPENAI_API_KEY
        )
        # Parser for combined validation and metadata extraction
        self.combined_parser = PydanticOutputParser(pydantic_object=ValidationAndMetadata)
        # Storage path for duplicate detection
        self.storage_dir = Path("data_storage_call_center")
        self.hashes_file = self.storage_dir / "transcript_hashes.json"
        self._ensure_hashes_file()
    
    def _ensure_hashes_file(self):
        """Ensure the transcript hashes file exists."""
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.hashes_file.exists():
            with open(self.hashes_file, 'w', encoding='utf-8') as f:
                json.dump({"hashes": []}, f)
    
    def _compute_transcript_hash(self, transcript: str) -> str:
        """
        Compute a SHA-256 hash of the transcript for duplicate detection.
        Normalizes the text before hashing to catch similar transcripts.
        
        Args:
            transcript: The transcript text
            
        Returns:
            Hex digest of the hash
        """
        # Normalize: lowercase, remove extra whitespace, strip
        normalized = ' '.join(transcript.lower().strip().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, transcript_hash: str) -> bool:
        """
        Check if a transcript hash already exists in the database.
        
        Args:
            transcript_hash: The hash to check
            
        Returns:
            True if duplicate, False otherwise
        """
        try:
            with open(self.hashes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return transcript_hash in data.get("hashes", [])
        except Exception:
            return False
    
    def _store_hash(self, transcript_hash: str):
        """
        Store a transcript hash in the database.
        
        Args:
            transcript_hash: The hash to store
        """
        try:
            with open(self.hashes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if transcript_hash not in data.get("hashes", []):
                data["hashes"].append(transcript_hash)
                
                with open(self.hashes_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not store transcript hash: {e}")
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the transcription, validate it, and extract metadata in a SINGLE LLM call.
        
        Args:
            state: Current agent state containing transcription
            
        Returns:
            Updated state with call_data or validation error
        """
        try:
            # Get transcript from state
            transcript = state.get("transcript", "")
            
            if not transcript:
                state["error"] = "No transcription available for processing"
                return state
            
            # Check for duplicate transcript FIRST (before any processing)
            transcript_hash = self._compute_transcript_hash(transcript)
            if self._is_duplicate(transcript_hash):
                state["error"] = "Duplicate transcript detected. This call has already been processed."
                state["duplicate_detected"] = True
                return {
                    **state,
                    "processing_steps": ["⚠️ Duplicate Detected: This transcript has already been processed. Skipping."]
                }
            
            # PRE-VALIDATION: Quick checks before calling LLM (saves API costs!)
            pre_validation_error = self._pre_validate_transcript(transcript)
            if pre_validation_error:
                state["error"] = pre_validation_error
                state["validation_failed"] = True
                return {
                    **state,
                    "processing_steps": [f"❌ Validation Failed: {pre_validation_error}"]
                }
            
            # COMBINED: Validate AND extract metadata in ONE LLM call (efficient!)
            validation_result = self._validate_and_extract_metadata(transcript)
            
            # Check validation result
            if not validation_result.is_valid:
                state["error"] = f"Not a valid call center conversation: {validation_result.validation_reason}"
                state["validation_failed"] = True
                return {
                    **state,
                    "processing_steps": [f"❌ Validation Failed: {validation_result.validation_reason}"]
                }
            
            # Valid! Use the extracted metadata
            metadata = validation_result.metadata or CallMetadata()
            
            # Extract conversation structure
            conversation, conversation_turns = self._extract_conversation(transcript)

            # Create CallData object
            call_data = CallData(
                metadata=metadata,
                conversation=conversation,
                conversation_turns=conversation_turns
            )
            
            # Store the transcript hash to prevent future duplicates
            self._store_hash(transcript_hash)
            
            # Update state - return new list for operator.add
            state["call_data"] = call_data
            
            return {
                **state,
                "processing_steps": ["Call Intake: Validated and extracted metadata"]
            }
            
        except Exception as e:
            state["error"] = f"Call Intake Agent error: {str(e)}"
            return state
    
    def _pre_validate_transcript(self, text: str) -> Optional[str]:
        """
        Pre-validate transcript using fast heuristics BEFORE calling LLM.
        This saves API costs by catching obvious invalid inputs early.
        
        Args:
            text: Transcript text to validate
            
        Returns:
            Error message if invalid, None if passes pre-validation
        """
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # 1. Length checks
        if len(text_stripped) < 50:
            return "Transcript too short (minimum 50 characters)"
        
        if len(text_stripped) > 100000:  # ~100K chars limit
            return f"Transcript too long ({len(text_stripped):,} characters). Maximum 100,000 characters allowed."
        
        # 2. Check for conversation indicators
        # Look for speaker labels or dialogue patterns
        conversation_indicators = [
            'agent:', 'caller:', 'customer:', 'representative:', 'client:',
            'support:', 'service:', 'operator:', 'rep:', 'csr:',
            # Also check for common dialogue patterns
            'agent -', 'caller -', 'customer -',
        ]
        
        has_conversation_pattern = any(indicator in text_lower for indicator in conversation_indicators)
        
        # Check for question marks (indicates dialogue)
        has_questions = '?' in text
        
        # Check for multiple sentences (basic dialogue check)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        if not has_conversation_pattern and sentence_count < 3:
            return "Does not appear to be a conversation transcript (no speaker labels or dialogue detected)"
        
        # 3. Check for obvious non-conversation content
        music_keywords = [
            '[music playing]', '[instrumental]', 'lyrics:', 'verse:', 'chorus:',
            '🎵', '♪', '♫', 'song:', 'album:', 'artist:'
        ]
        
        if any(keyword in text_lower for keyword in music_keywords):
            return "Appears to be music or lyrics, not a call center conversation"
        
        # 4. Check for gibberish (excessive special characters)
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / len(text) if len(text) > 0 else 0
        
        if special_char_ratio > 0.3:  # More than 30% special characters
            return "Transcript contains excessive special characters or appears to be gibberish"
        
        # 5. Check for minimum word count (actual dialogue should have substantial content)
        word_count = len(text_stripped.split())
        
        if word_count < 20:
            return f"Transcript too short ({word_count} words). Minimum 20 words required."
        
        # 6. Check for repeated patterns (bot/spam detection)
        words = text_stripped.split()
        if len(words) > 10:
            unique_words = set(words)
            uniqueness_ratio = len(unique_words) / len(words)
            
            if uniqueness_ratio < 0.3:  # Less than 30% unique words
                return "Transcript appears to be repetitive or spam"
        
        # 7. Check character encoding (detect binary/corrupted files)
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            return "Transcript contains invalid characters or encoding issues"
        
        # Passed all pre-validation checks!
        return None
    
    def _validate_and_extract_metadata(self, text: str) -> ValidationAndMetadata:
        """
        COMBINED: Validate if transcript is a call center conversation AND extract metadata.
        This uses a SINGLE LLM call instead of two separate calls (50% cost savings!).
        
        Args:
            text: Transcription text
            
        Returns:
            ValidationAndMetadata object with validation result and extracted metadata
        """
        prompt_template = ChatPromptTemplate.from_template(
            """You are a call center quality assurance expert with two tasks:

TASK 1 - VALIDATION: Determine if this is a legitimate call center conversation
TASK 2 - EXTRACTION: If valid, extract the call metadata

TEXT TO ANALYZE:
{text}

VALIDATION CRITERIA:
✓ VALID if:
  - Conversation between agent and customer
  - Customer service/support context
  - Actual dialogue with questions and responses
  - Business or service interaction

✗ INVALID if:
  - Music lyrics or instrumental sounds
  - Random audio/noises/non-speech
  - Single person monologue (no customer interaction)
  - Gibberish or nonsensical text
  - Poetry, stories, creative writing
  - Technical descriptions ("music playing", "background noise")

METADATA TO EXTRACT (if valid):
1. Call ID - Any identifier (Call ID, Reference Number, Ticket ID, etc.)
2. Caller Name - Customer's name
3. Agent Name - Service agent's name
4. Call Duration - How long the call lasted (any format: "5:23", "5 minutes", etc.)
5. Date/Time - When the call occurred

IMPORTANT: If any metadata field cannot be found, set it to null.

{format_instructions}

Respond with structured JSON containing:
- is_valid: true/false
- validation_reason: Brief explanation
- metadata: Extracted fields (or null if invalid)
"""
        )
        
        format_instructions = self.combined_parser.get_format_instructions()
        
        messages = prompt_template.format_messages(
            text=text[:2000],  # Use first 2000 chars for efficiency
            format_instructions=format_instructions
        )
        
        try:
            response = self.llm.invoke(messages)
            result = self.combined_parser.parse(response.content)
            return result
        except Exception as e:
            # If parsing fails, return invalid with error message
            print(f"Combined validation/extraction failed: {e}")
            return ValidationAndMetadata(
                is_valid=False,
                validation_reason=f"Unable to validate transcript: {str(e)}",
                metadata=None
            )
    
    def _extract_conversation(self, text: str) -> tuple[str, List[ConversationTurn]]:
        """
        Extract and parse conversation using LLM for intelligent parsing.
        
        Args:
            text: Transcription text
            
        Returns:
            Tuple of (full conversation string, list of conversation turns)
        """
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert at parsing call center conversations.
Analyze the following text and extract the conversation between the agent and caller.

TEXT:
{text}

Please:
1. Identify where the actual conversation starts (ignoring metadata like Call ID, dates, etc.)
2. Parse the conversation into individual turns
3. Identify the speaker for each turn (Agent, Caller, Customer, Representative, etc.)
4. Extract what was said by each speaker

Return a JSON object with:
- "conversation": the full conversation text
- "turns": an array of {{"speaker": "role", "text": "what they said"}}

Be flexible in recognizing different conversation formats. Speakers might be labeled as:
- Agent, Representative, Customer Service, Support Agent, etc.
- Caller, Customer, Client, etc.

Example output:
{{
  "conversation": "Agent: Hello...\\nCaller: Hi...",
  "turns": [
    {{"speaker": "Agent", "text": "Hello, how can I help?"}},
    {{"speaker": "Caller", "text": "I have a problem..."}}
  ]
}}

Only return valid JSON, no additional text.
"""
        )
        
        messages = prompt_template.format_messages(text=text)
        
        try:
            response = self.llm.invoke(messages)
            # Parse the JSON response
            result = json.loads(response.content)
            
            conversation_text = result.get("conversation", "")
            turns_data = result.get("turns", [])
            
            # Convert to ConversationTurn objects
            conversation_turns = [
                ConversationTurn(
                    speaker=turn.get("speaker", "Unknown"),
                    text=turn.get("text", "")
                )
                for turn in turns_data
            ]
            
            return conversation_text, conversation_turns
            
        except Exception as e:
            print(f"LLM conversation extraction failed: {e}. Returning raw text.")
            # Return the raw text as fallback
            return text, []
    
    def validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate that the input text contains content.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or len(text.strip()) < 10:
            return False, "Input text is too short or empty"
        
        # Basic validation - just check if there's enough content
        # LLM will handle flexible conversation detection
        return True, None
