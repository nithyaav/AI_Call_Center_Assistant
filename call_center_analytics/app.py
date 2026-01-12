"""
Enhanced Streamlit App for Call Center AI Assistant with Data Storage
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
from agents.workflow import CallCenterWorkflow
from agents.data_storage_agent import DataStorageAgent
from utils.config import Config
from utils.models import CallData, CallSummary, QualityScore


# Page configuration
st.set_page_config(
    page_title="Call Center AI Assistant",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #17a2b8;
        font-weight: bold;
    }
    .score-average {
        color: #ffc107;
        font-weight: bold;
    }
    .score-poor {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Customize metadata metric font sizes */
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
        font-weight: 600;
    }
    
    /* Make sidebar navigation more prominent */
    .stRadio > label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1f77b4 !important;
    }
    
    /* Style radio buttons to look like navigation tabs */
    .stRadio > div {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stRadio > div > label {
        padding: 0.75rem 1rem !important;
        border-radius: 0.25rem !important;
        margin: 0.25rem 0 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .stRadio > div > label:hover {
        background-color: #e0e5eb !important;
    }
    
    /* Sidebar title styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Make expander headers more subtle */
    .streamlit-expanderHeader {
        font-size: 0.95rem !important;
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)


def get_score_class(score: float) -> str:
    """Get CSS class based on score value."""
    if score >= 8.5:
        return "score-excellent"
    elif score >= 7.0:
        return "score-good"
    elif score >= 5.0:
        return "score-average"
    else:
        return "score-poor"


def get_score_label(score: float) -> str:
    """Get label based on score value."""
    if score >= 8.5:
        return "Excellent"
    elif score >= 7.0:
        return "Good"
    elif score >= 5.0:
        return "Average"
    else:
        return "Needs Improvement"


def display_metadata(call_data: CallData):
    """Display call metadata."""
    st.subheader("📋 Call Metadata")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Call ID", call_data.metadata.call_id or "N/A")
    
    with col2:
        st.metric("Caller", call_data.metadata.caller_name or "N/A")
    
    with col3:
        st.metric("Agent", call_data.metadata.agent_name or "N/A")
    
    with col4:
        st.metric("Duration", call_data.metadata.call_duration or "N/A")
    
    if call_data.metadata.date_time:
        st.info(f"📅 Call Date: {call_data.metadata.date_time}")


def display_summary(summary: CallSummary):
    """Display call summary."""
    st.subheader("📝 Call Summary")
    
    st.markdown(f"**{summary.brief_summary}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔑 Key Points")
        for point in summary.key_points:
            st.markdown(f"• {point}")
    
    with col2:
        st.markdown("### 🎯 Action Items")
        if summary.action_items:
            for item in summary.action_items:
                st.markdown(f"✓ {item}")
        else:
            st.info("No action items identified")
    
    if summary.customer_issue:
        st.markdown("### ❓ Customer Issue")
        st.write(summary.customer_issue)
    
    if summary.resolution:
        st.markdown("### ✅ Resolution")
        st.success(summary.resolution)


def display_quality_score(quality_score: QualityScore):
    """Display quality scores."""
    st.subheader("⭐ Quality Evaluation")
    
    # Overall score (prominent)
    score_class = get_score_class(quality_score.overall_score)
    score_label = get_score_label(quality_score.overall_score)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; border-radius: 0.5rem;">
        <h2>Overall Score</h2>
        <h1 class="{score_class}">{quality_score.overall_score:.1f}/10</h1>
        <h3 class="{score_class}">{score_label}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Individual scores
    st.markdown("### 📊 Detailed Scores")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tone & Empathy",
            f"{quality_score.tone_score:.1f}/10",
            delta=None
        )
    
    with col2:
        st.metric(
            "Professionalism",
            f"{quality_score.professionalism_score:.1f}/10",
            delta=None
        )
    
    with col3:
        st.metric(
            "Resolution/Effectiveness",
            f"{quality_score.resolution_score:.1f}/10",
            delta=None
        )
    
    with col4:
        st.metric(
            "Response Time",
            f"{quality_score.response_time_score:.1f}/10",
            delta=None
        )
    
    st.markdown("---")
    
    # Feedback
    if quality_score.feedback:
        st.markdown("### 💬 Feedback")
        st.info(quality_score.feedback)
    
    # Strengths and improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✨ Strengths")
        for strength in quality_score.strengths:
            st.success(f"✓ {strength}")
    
    with col2:
        st.markdown("### 🎯 Areas for Improvement")
        for improvement in quality_score.areas_for_improvement:
            st.warning(f"⚠ {improvement}")


def display_conversation(call_data: CallData):
    """Display conversation transcript."""
    st.subheader("💬 Full Conversation")
    
    if call_data.conversation_turns:
        for turn in call_data.conversation_turns:
            if turn.speaker.lower() == "agent":
                st.markdown(f"**🎧 Agent:** {turn.text}")
            else:
                st.markdown(f"**👤 {turn.speaker}:** {turn.text}")
            st.markdown("---")
    else:
        with st.expander("View Full Transcript"):
            st.text(call_data.conversation)


def process_call_page():
    """Page for processing new calls."""
    # Page header with icon - compact version
    st.markdown("### 📞 Process Call")
    st.caption("Upload audio files or paste text transcripts to analyze call center interactions.")
    
    # File upload
    upload_tab, text_tab = st.tabs(["Upload File", "Paste Text"])
    
    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a call recording (audio) or transcript (text)",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'txt'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG for audio; TXT for transcripts"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file)
    
    with text_tab:
        text_input = st.text_area(
            "Paste call transcript",
            height=300,
            placeholder="""Example format:
Call ID: 12345
Date: 2026-01-05 14:30:00
Duration: 5:23
Caller Name: John Doe
Agent Name: Jane Smith

Conversation:
Agent: Thank you for calling. How may I help you today?
Caller: I have an issue with my recent order.
..."""
        )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                process_text_input(text_input)
            else:
                st.warning("Please paste some text to analyze.")


def agent_performance_page():
    """Page for viewing agent performance."""
    # Page header with icon
    st.markdown("## 👥 Agent Performance")
    st.caption("View analytics, rankings, and detailed reports for call center agents.")
    st.markdown("---")
    
    # Initialize storage agent
    storage_agent = DataStorageAgent()
    
    # Get performance data and manual review data
    performance_df = storage_agent.get_agent_performance()
    manual_review_count = len(storage_agent.get_manual_review_calls())
    manual_review_label = f"🔍 Manual Review Queue ({manual_review_count})" if manual_review_count > 0 else "🔍 Manual Review Queue"
    
    # Check if we have ANY data (performance or manual review)
    has_performance_data = not performance_df.empty
    has_manual_review_data = manual_review_count > 0
    
    if not has_performance_data and not has_manual_review_data:
        st.info("📊 No data available yet. Process some calls to see agent performance and reviews.")
        return
    
    # Display summary metrics (only if we have performance data)
    if has_performance_data:
        st.subheader("📈 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", len(performance_df))
        
        with col2:
            st.metric("Total Calls", int(performance_df['total_calls'].sum()))
        
        with col3:
            avg_score = performance_df['average_overall_score'].mean()
            st.metric("Average Score", f"{avg_score:.2f}/10")
        
        with col4:
            top_agent = performance_df.iloc[0]['agent_name']
            st.metric("Top Performer", top_agent)
        
        st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🏆 Agent Rankings", "📋 Agent Reports", manual_review_label])
    
    # TAB 1: Agent Rankings
    with tab1:
        st.subheader("🏆 Agent Performance Rankings")
        
        if not has_performance_data:
            st.info("""
            📊 **No agent rankings available yet.**
            
            Rankings appear when agents are identified and scored in call transcripts.
            
            **Why might this be empty?**
            - No calls with identified agents have been processed yet
            - All processed calls are flagged for manual review (no agent names identified)
            - Calls with content safety violations don't generate agent scores
            
            **To see rankings:** Process calls with clear agent identification in the transcript.
            """)
        else:
            # Format the dataframe for display
            display_df = performance_df.copy()
            
            # Add rank column starting from 1
            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
            
            display_df['average_overall_score'] = display_df['average_overall_score'].round(2)
            display_df['average_tone_score'] = display_df['average_tone_score'].round(2)
            display_df['average_professionalism_score'] = display_df['average_professionalism_score'].round(2)
            display_df['average_resolution_score'] = display_df['average_resolution_score'].round(2)
            display_df['average_response_score'] = display_df['average_response_score'].round(2)
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'agent_name': 'Agent Name',
                'total_calls': 'Total Calls',
                'average_overall_score': 'Overall Score',
                'average_tone_score': 'Tone',
                'average_professionalism_score': 'Professional',
                'average_resolution_score': 'Resolution',
                'average_response_score': 'Response',
                'last_updated': 'Last Updated'
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Add explanation
            st.info("""
            **About Rankings:**
            - Agents are ranked by **Overall Score** (highest first)
            - Only calls with complete data are included in rankings
            - Calls needing manual review are excluded to ensure accuracy
            """)
    
    # TAB 2: Agent Reports
    with tab2:
        st.subheader("📋 Individual Agent Report")
        
        if not has_performance_data:
            st.info("""
            📋 **No agent reports available yet.**
            
            Individual agent reports are generated once agents are identified and scored.
            
            Process calls with agent names to see detailed performance reports.
            """)
        else:
            agent_names = performance_df['agent_name'].tolist()
            selected_agent = st.selectbox("Select Agent", agent_names, key="agent_report_select")
            
            if selected_agent:
                report = storage_agent.generate_agent_report(selected_agent)
                
                if "error" in report:
                    st.error(report["error"])
                else:
                    # Display report
                    st.markdown(f"### Agent: **{selected_agent}**")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Calls", report['total_calls'])
                    
                    with col2:
                        st.metric("Average Score", f"{report['scores']['overall']['average']:.2f}/10")
                    
                    with col3:
                        st.metric("Performance Rating", report['performance_rating'])
                    
                    st.markdown("---")
                    
                    # Score breakdown
                    st.markdown("### 📊 Score Breakdown")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        score = report['scores']['tone']
                        st.metric(
                            "Tone & Empathy", 
                            f"{score['average']:.2f}",
                            delta=score['trend'].replace('_', ' ').title()
                        )
                    
                    with col2:
                        score = report['scores']['professionalism']
                        st.metric(
                            "Professionalism", 
                            f"{score['average']:.2f}",
                            delta=score['trend'].replace('_', ' ').title()
                        )
                    
                    with col3:
                        score = report['scores']['resolution']
                        st.metric(
                            "Resolution/Effectiveness", 
                            f"{score['average']:.2f}",
                            delta=score['trend'].replace('_', ' ').title()
                        )
                    
                    with col4:
                        score = report['scores']['response']
                        st.metric(
                            "Response", 
                            f"{score['average']:.2f}",
                            delta=score['trend'].replace('_', ' ').title()
                        )
                    
                    # Recent calls
                    st.markdown("### 📞 Recent Calls")
                    recent_df = pd.DataFrame(report['recent_calls'])
                    if not recent_df.empty:
                        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                        recent_df['overall_score'] = recent_df['overall_score'].round(2)
                        st.dataframe(recent_df, use_container_width=True)
    
    # TAB 3: Manual Review Queue
    with tab3:
        st.subheader("🔍 Manual Review Queue")
        
        manual_review_calls = storage_agent.get_manual_review_calls()
        
        if not manual_review_calls:
            st.success("✅ No calls pending manual review! All processed calls have complete data.")
        else:
            st.warning(f"⚠️ **{len(manual_review_calls)} call(s)** flagged for manual review")
            
            st.markdown("""
            **Calls need manual review when:**
            - 🚨 **Content Safety Violation**: Inappropriate content detected by guardrails
            - ⚠️ **Missing Agent Name**: No agent was identified in the conversation
            - ⚠️ **Incomplete Scoring**: Quality scoring was incomplete (< 2 scores extracted)
            
            Flagged calls are **saved with full data** but require human review before further processing.
            """)
            
            # Display manual review calls
            for idx, call in enumerate(manual_review_calls, 1):
                # Check if this is flagged content (from content safety)
                is_flagged_content = call.get('status') == 'FLAGGED_FOR_REVIEW'
                call_id = call.get('call_id', 'Unknown')
                timestamp = call.get('timestamp', '')[:19] if call.get('timestamp') else 'Unknown'
                
                # Set title based on type
                if is_flagged_content:
                    title = f"🚨 FLAGGED: {call_id} - {timestamp}"
                else:
                    title = f"📋 Call {idx}: {call_id} - {timestamp[:10]}"
                
                with st.expander(title):
                    # Show flagged content differently
                    if is_flagged_content:
                        st.error("🚨 **CONTENT SAFETY VIOLATION**")
                        
                        # Show flagged categories
                        flagged_categories = call.get('flagged_categories', [])
                        if flagged_categories:
                            st.warning(f"**Flagged Categories:** {', '.join(flagged_categories)}")
                        
                        # Show metadata
                        st.markdown("**Status:** Awaiting manual review")
                        st.markdown(f"**Input Type:** {call.get('input_type', 'unknown')}")
                        st.markdown(f"**Timestamp:** {timestamp}")
                        
                        # Show transcript
                        transcript = call.get('transcript', '')
                        if transcript:
                            st.markdown("**Transcript:**")
                            st.text_area("Transcript Content", transcript, height=300, disabled=True, key=f"transcript_{call_id}")
                        
                        st.error("⚠️ **Action Required:** Human review needed before proceeding")
                    
                    else:
                        # Normal call needing review (missing agent name or incomplete scoring)
                        # Metadata
                        metadata = call.get('metadata', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            agent_name = metadata.get('agent_name', 'N/A')
                            if not agent_name or not agent_name.strip():
                                st.markdown("**Agent:** ⚠️ Not Identified")
                            else:
                                st.markdown(f"**Agent:** {agent_name}")
                        
                        with col2:
                            caller_name = metadata.get('caller_name', 'N/A')
                            st.markdown(f"**Caller:** {caller_name}")
                        
                        with col3:
                            duration = metadata.get('call_duration', 'N/A')
                            st.markdown(f"**Duration:** {duration}")
                        
                        # Summary
                        summary = call.get('summary')
                        if summary:
                            st.markdown("**Summary:**")
                            st.info(summary.get('brief_summary', 'No summary available'))
                            
                            if summary.get('customer_issue'):
                                st.markdown(f"**Issue:** {summary['customer_issue']}")
                            
                            if summary.get('resolution'):
                                st.markdown(f"**Resolution:** {summary['resolution']}")
                        
                        # Quality Score (if available)
                        quality_score = call.get('quality_score')
                        if quality_score:
                            st.markdown("**Quality Score (Incomplete):**")
                            score_col1, score_col2, score_col3 = st.columns(3)
                            with score_col1:
                                if quality_score.get('overall_score'):
                                    st.metric("Overall", f"{quality_score['overall_score']:.1f}/10")
                            with score_col2:
                                if quality_score.get('tone_score'):
                                    st.metric("Tone", f"{quality_score['tone_score']:.1f}/10")
                            with score_col3:
                                if quality_score.get('professionalism_score'):
                                    st.metric("Professional", f"{quality_score['professionalism_score']:.1f}/10")
                        else:
                            st.markdown("**Quality Score:** ⚠️ Not Available")
                        
                        # Show reason for manual review
                        if not metadata.get('agent_name') or not metadata.get('agent_name', '').strip():
                            st.error("🚨 **Reason:** No agent name identified in conversation")
                        elif not quality_score:
                            st.error("🚨 **Reason:** Quality scoring failed or incomplete")
                        else:
                            st.error("🚨 **Reason:** Quality scoring incomplete (< 2 criteria extracted)")


def process_uploaded_file(uploaded_file):
    """Process uploaded file."""
    # Determine file type
    file_ext = Path(uploaded_file.name).suffix.lower()
    
    if file_ext == '.txt':
        # Text file
        input_type = "text"
        input_content = uploaded_file.read().decode('utf-8')
    elif file_ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
        # Audio file
        input_type = "audio"
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_content = tmp_file.name
    else:
        st.error(f"Unsupported file format: {file_ext}")
        return
    
    # Process the input
    process_input(input_type, input_content, uploaded_file.name)
    
    # Clean up temp file
    if input_type == "audio":
        try:
            os.unlink(input_content)
        except:
            pass


def process_text_input(text_content: str):
    """Process text input."""
    process_input("text", text_content, "pasted_text")


def process_input(input_type: str, input_content: str, filename: str):
    """Process input through the workflow."""
    
    st.markdown("---")
    st.header("🔄 Processing")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize workflow
        status_text.text("Initializing workflow...")
        progress_bar.progress(10)
        
        workflow = CallCenterWorkflow()
        
        # Process input
        status_text.text(f"Processing {input_type} input...")
        progress_bar.progress(30)
        
        result = workflow.process(input_type, input_content)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Check for duplicate detection first
        if result.get("duplicate_detected"):
            progress_bar.empty()
            status_text.empty()
            
            st.warning("⚠️ Duplicate Transcript Detected")
            
            st.info(
                "This transcript has already been processed and stored in the system.\n\n"
                "**Why avoid duplicates?**\n"
                "- Prevents duplicate records in the database\n"
                "- Avoids skewing agent performance metrics\n"
                "- Saves API costs by not reprocessing the same content\n\n"
                "**If you need to reprocess this call:**\n"
                "- Make edits to the transcript (even minor changes)\n"
                "- Or contact an administrator to clear the duplicate detection cache"
            )
            
            return
        
        # Check for validation failure
        if result.get("validation_failed"):
            progress_bar.empty()
            status_text.empty()
            
            st.error("❌ Invalid Input Detected")
            
            st.warning(
                f"**{result.get('error', 'Not recognized as a call center conversation')}**\n\n"
                "This system is designed to analyze call center conversations between agents and customers. "
                "Please ensure your input contains:\n"
                "- A dialogue between two or more people\n"
                "- Customer service or support context\n"
                "- Questions, responses, and interaction\n\n"
                "**Not supported:** Music files, monologues, random audio, or non-conversation content."
            )
            
            return
        
        # Check for content safety failures
        if not result.get("content_safety_passed", True):
            progress_bar.empty()
            status_text.empty()
            
            flagged_categories = result.get("content_safety_details", {}).get("flagged_categories", [])
            
            st.error("Content Safety Alert")
            
            st.warning(
                f"**This content has been flagged as inappropriate and stored for manual review.**\n\n"
                f"**Flagged Categories:** {', '.join(flagged_categories) if flagged_categories else 'Unknown'}\n\n"
                f"**Status:** Saved for manual review - Will not proceed through normal processing\n\n"
            )
            
            # Show where it was stored
            if result.get("storage_path"):
                st.info(f"Stored at: {result['storage_path']}")
                
                # Show processing steps
                if result.get("processing_steps"):
                    with st.expander("View Processing Details", expanded=False):
                        for step in result.get("processing_steps", []):
                            st.write(f"• {step}")
            
            return
        
        # Check for other errors
        if result.get("error"):
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {result['error']}")
            
            return
        
        # Display processing steps
        with st.expander("View Processing Steps", expanded=False):
            for step in result.get("processing_steps", []):
                # Check if it's a warning step (starts with ⚠️)
                if step.startswith("⚠️"):
                    st.warning(step)
                else:
                    st.success(f"✓ {step}")
        
        # Show storage confirmation or warning
        if result.get("storage_path"):
            st.success(f"✅ Data saved to: {result['storage_path']}")
        else:
            # Check if storage was skipped
            storage_steps = [s for s in result.get("processing_steps", []) if "Data Storage" in s]
            if storage_steps and any("Skipped" in s for s in storage_steps):
                st.warning("⚠️ Call data was not saved due to missing essential metadata (agent name)")
        
        st.markdown("---")
        
        # Display results
        st.header("📊 Analysis Results")
        
        # Metadata
        if result.get("call_data"):
            display_metadata(result["call_data"])
            st.markdown("---")
        
        # Summary
        if result.get("summary"):
            display_summary(result["summary"])
            st.markdown("---")
        
        # Check if call needs manual review (could be due to scoring failure or missing agent)
        if result.get("needs_manual_review"):
            # Check if it's due to missing agent name
            call_data = result.get("call_data")
            if call_data and (not call_data.metadata.agent_name or not call_data.metadata.agent_name.strip()):
                st.warning("⚠️ **Manual Review Required**: No agent name was identified in this call. The call has been saved for manual review and will not be included in agent performance rankings.")
            else:
                st.warning("⚠️ **Manual Review Required**: Quality scoring was incomplete for this call. The call has been saved for manual review and will not be included in agent performance rankings.")
            st.markdown("---")
        
        # Quality Score (may not exist if agent name was missing or scoring failed)
        if result.get("quality_score"):
            display_quality_score(result["quality_score"])
            st.markdown("---")
        else:
            # Check if quality scoring was skipped or failed
            quality_steps = [s for s in result.get("processing_steps", []) if "Quality Scoring" in s]
            storage_steps = [s for s in result.get("processing_steps", []) if "Data Storage" in s]
            
            if quality_steps:
                if any("Failed" in s or "Error" in s for s in quality_steps):
                    # Scoring failed - already showed warning above
                    pass
                elif any("Skipped" in s for s in quality_steps):
                    st.info("ℹ️ Quality scoring was skipped because no agent name was identified in the call.")
                    st.markdown("---")
            
            # Show that data was still saved
            if storage_steps and any("saved for manual review" in s.lower() for s in storage_steps):
                st.success("✅ Call data and transcript have been saved for later review.")
                st.markdown("---")
        
        # Conversation (at the end)
        if result.get("call_data"):
            display_conversation(result["call_data"])
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">AI Call Center Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar with Navigation at Top
    with st.sidebar:
        # App Title
        st.title("📞 Call Center AI Assistant")
        st.markdown("### Navigation")
        
        # Navigation - Prominent at the top
        page = st.radio(
            label="Select Page",
            options=["📞 Process Call", "👥 Agent Performance"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # About Section - Collapsible
        with st.expander("ℹ️ About System", expanded=False):
            st.info("""
            Call Center AI Assistant - AI-powered analytics using multi-agent system:
            
            - **Transcription**: Audio to text (OpenAI Whisper)
            - **Content Safety**: Inappropriate content detection (OpenAI Moderation API)
            - **Intake**: Metadata extraction & validation
            - **Summarization**: AI-generated summaries & key points
            - **Quality Scoring**: Performance evaluation (4 dimensions)
            - **Data Storage**: Complete persistence & analytics
            """)
        
        # Settings Section - Collapsible
        with st.expander("⚙️ System Info", expanded=False):
            st.write(f"**GPT Model:** {Config.GPT_MODEL}")
            st.write(f"**Whisper:** {Config.WHISPER_MODEL}")
            
            # Check API key
            if Config.OPENAI_API_KEY and Config.OPENAI_API_KEY != "your_openai_api_key_here":
                st.success("✅ API Configured")
            else:
                st.error("❌ API Key Missing")
                st.warning("Set OPENAI_API_KEY in .env")
    
    st.markdown("---")
    
    if page == "📞 Process Call":
        process_call_page()
    elif page == "👥 Agent Performance":
        agent_performance_page()


if __name__ == "__main__":
    main()
