# Call Center AI Assistant 📞

A sophisticated multi-agent system for automated call center analytics using **LangGraph**, **LangChain**, **OpenAI GPT-4**, and **Streamlit**. Features persistent data storage, agent performance tracking, and quality analytics.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.45-green.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.9-orange.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-red.svg)](https://streamlit.io/)

</div>

## 🎯 Business Use Case

In modern call centers, crucial insights from conversations are often trapped in long transcripts or voice recordings. Manual summaries, quality checks, and agent evaluations are time-consuming and inconsistent.

**This system provides:**
- 🤖 **Automated Insight Extraction**: Rapidly summarize calls without manual effort
- 📊 **QA Monitoring at Scale**: Score service quality using LLM agents
- 👥 **Agent Performance Tracking**: Monitor and evaluate agents over time
- 📈 **Data-Driven Decisions**: Use analytics for promotions and training
- ✅ **Consistency & Compliance**: Standardize evaluations across interactions
- 🎤 **Voice-to-Insights Pipeline**: Convert audio into structured data for decision-making

## ✨ Features

### �️ **Content Safety Agent** ✨ 
- **Automated content moderation** using OpenAI Moderation API
- Checks transcribed text for inappropriate content:
  - Harassment and threatening language
  - Hate speech
  - Violence and graphic content
  - Sexual content
  - Self-harm content
- Flags violations and routes to manual review
- Stores flagged content with FLAGGED_ prefix for easy identification
- Preserves full transcript and metadata for compliance

### �🔍 **Call Intake Agent** ✨ 
- **Combined Validation & Metadata Extraction** (50% API cost reduction!)
- Validates input formats and extracts metadata in a single LLM call
- Intelligently parses caller name, agent name, call duration, call ID, date/time
- Structures entire conversation into speaker turns
- **No regex patterns** - pure LLM-based extraction for flexibility
- Graceful handling of missing agent names (saves for manual review)

### 🎤 **Transcription Agent**
- Converts audio to text using OpenAI Whisper
- Supports multiple audio formats (WAV, MP3, M4A, FLAC, OGG)
- Handles files up to 25 MB
- Output fed directly to Content Safety check

### 📝 **Summarization Agent**
- Generates concise summaries and key points using GPT-4
- Identifies customer issues and resolutions
- Extracts action items and follow-ups
- **Intelligent routing**: Determines if quality scoring should proceed

### ⭐ **Quality Scoring Agent** ✨ 
- Evaluates calls on 4 key dimensions:
  - Tone & Empathy (0-10)
  - Professionalism (0-10)
  - Problem Resolution or Call Effectiveness (0-10)
  - Response Appropriateness (0-10)
- **Context-Aware Scoring**: Adapts rubric for problem resolution vs informational calls
- **Smart Fallback Scoring**: 
  - Calculates mean only from successfully extracted scores
  - No arbitrary defaults that bias averages
  - Flags incomplete scoring for manual review
- Provides detailed feedback with strengths and improvement areas
- **Always runs**: Provides insights even when agent name is missing

### 💾 **Data Storage Agent** ✨ Complete Persistence
- **ALWAYS SAVES DATA** - Zero data loss guarantee!
- Persists ALL calls regardless of:
  - Content safety status (flagged content stored separately)
  - Agent name presence (saves with `needs_manual_review` flag)
  - Quality scoring status (saves even if scoring fails)
- Stores data in multiple formats (JSON, CSV)
- **Selective Analytics Updates**: Only updates agent rankings when attribution is reliable
- Tracks agent performance over time with full audit trail
- Generates comprehensive agent reports with trends
- Supports performance reviews and promotion decisions
- **Manual Review System**: Flags incomplete/unsafe calls for human verification
- **Flagged Content Storage**: Separate files with FLAGGED_ prefix for content safety violations

### 🔄 **Workflow Orchestration (LangGraph)** ✨ Enhanced
- Orchestrates workflow between all 6 agents
- **Intelligent Conditional Routing**:
  - Audio vs text input routing
  - Content safety check after transcription
  - Conditional edge: flagged content routes to storage (not blocked)
  - Quality scoring always runs (provides insights even without agent name)
  - **Always routes to storage** - no data loss
- Error handling and state management
- Optimized with `operator.add` for processing steps
- **Cost Optimizations**: Combined validation + extraction (35% reduction)

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Navigate to project directory
cd call_center_analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Copy the example file and add your OpenAI API key
cp .env.example .env
# Then edit .env and replace 'your_openai_api_key_here' with your actual key

# Verify setup
python test_setup.py

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`


## 💻 Usage

### Two-Page Interface

#### 📞 **Process Call Page**
1. **Upload File Tab**: Upload audio (.wav, .mp3, .m4a, .flac, .ogg) or text (.txt) files
2. **Paste Text Tab**: Directly paste call transcripts
3. **Last Result Tab** ⭐ NEW!: View your most recent analysis (persists across navigation)

#### 👥 **Agent Performance Page** ⭐ ENHANCED!
1. **Overall Statistics**: View total agents, calls, and average scores (when agents are ranked)
2. **Agent Rankings Tab**: See all agents ranked by performance
   - Shows empty state with helpful message when no agents ranked
   - Excludes calls needing manual review for accuracy
3. **Agent Reports Tab**: Generate detailed reports with:
   - Total calls processed
   - Average scores across all categories
   - Performance trends (improving/declining/stable)
   - Performance rating (Outstanding to Unsatisfactory)
   - Recent call history
   - Shows empty state when no agent data available
4. **Manual Review Queue Tab** 🆕: Review flagged content
   - Badge shows count of items needing review
   - Three types of flagged items:
     - 🚨 **Content Safety Violations**: Inappropriate content detected
     - ⚠️ **Missing Agent Name**: No agent identified
     - ⚠️ **Incomplete Scoring**: Quality scoring failed
   - Shows full transcript for flagged content
   - Displays flagged categories for content safety violations
   - **Always accessible** even when no agents are ranked

### Processing a Call

**Option 1: Upload Files**
1. Navigate to "📞 Process Call" page
2. Click "Upload File" tab
3. Upload either:
   - **Text file** (.txt) with call transcript
   - **Audio file** (.wav, .mp3, .m4a, .flac, .ogg)

**Option 2: Paste Text**
1. Navigate to "📞 Process Call" page
2. Click "Paste Text" tab
3. Paste your call transcript directly
4. Click "Analyze Text"

**Option 3: View Last Result** 
1. Navigate to "📞 Process Call" page
2. Click "Last Result" tab
3. View your previous analysis (even after switching pages)

### View Results
The system displays:
- �️ **Content Safety Check**: Notification if content flagged for manual review
- �📋 **Call Metadata**: ID, names, duration, date/time
- 💬 **Conversation Transcript**: Full dialogue with speaker turns
- 📝 **AI-Generated Summary**: Key points, issues, resolution, and action items
- ⭐ **Quality Scores**: Detailed evaluation with feedback, strengths, and areas for improvement
- ✅ **Storage Confirmation**: Data automatically saved for future analytics
- ⚠️ **Manual Review Flags**: Alerts for content safety violations, missing agent, or incomplete scoring (data still saved!)

## 📁 Project Structure

```
call_center_analytics/
├── agents/                      # Agent implementations
│   ├── __init__.py             # Package initialization
│   ├── content_safety_agent.py # Content moderation (NEW!)
│   ├── call_intake_agent.py    # LLM-based metadata extraction
│   ├── transcription_agent.py  # Audio transcription with Whisper
│   ├── summarization_agent.py  # GPT-4 summary generation
│   ├── quality_scoring_agent.py # Quality assessment
│   ├── data_storage_agent.py   # Data persistence with flagged content handling
│   └── workflow.py             # LangGraph orchestration with content safety
├── utils/                       # Utilities
│   ├── config.py               # Configuration management
│   ├── models.py               # Pydantic data models
│   ├── guardrails.py           # Content safety guardrails (NEW!)
│   └── guardrails_config.py    # Guardrails configuration (NEW!)
├── data_storage_call_center/  # Persistent storage (auto-created)
│   ├── calls/                  # Individual call JSON files
│   │   ├── CALL_*.json        # Normal calls with agent attribution
│   │   └── FLAGGED_*.json     # Content safety violations (NEW!)
│   ├── reports/                # Agent performance reports
│   ├── calls_database.json     # Master call index with needs_manual_review flags
│   ├── quality_scores.csv      # All quality scores
│   ├── agent_performance.csv   # Agent statistics (excludes manual review calls)
│   └── transcript_hashes.json  # Duplicate detection (NEW!)
├── sample_data/                # Sample transcripts and audio
│   ├── example_transcript_good.txt         # High quality call
│   ├── example_transcript_excellent.txt    # Outstanding service
│   ├── example_transcript_poor.txt         # Poor quality call
│   ├── example_transcript_flagged.txt      # Content safety violation (NEW!)
│   └── example_transcript_abusive.txt      # Abusive language example (NEW!)
├── app.py                      # Streamlit web interface with Manual Review
├── requirements.txt            # Python dependencies
└── test_setup.py              # Setup verification script
```

## 📝 Input Format

### Text File Format
```
Call ID: CS-2026-001
Date: 2026-01-05 14:30:00
Duration: 5:23
Caller Name: John Doe
Agent Name: Sarah Johnson

Conversation:
Agent: Thank you for calling. How may I help you today?
Caller: I have an issue with my recent order.
Agent: I'm sorry to hear that...
```

### Audio Files
- Supported: WAV, MP3, M4A, FLAC, OGG
- Max size: 25 MB
- Automatically transcribed

## 🧪 Testing

Test with provided sample data:

```bash
# Run setup verification
python test_setup.py

# Start the app (recommended multi-page version)
streamlit run pages_app.py

# Then upload: sample_data/sample_call_transcript.txt
```

Two sample files provided:
- `sample_call_transcript.txt` - High quality call example
- `sample_call_poor_quality.txt` - Poor quality call example

### Testing Features
1. **Process a normal call** to see metadata extraction, summarization, and quality scoring
2. **Process flagged content** (use `example_transcript_flagged.txt`) to see content safety in action
3. **Switch to Agent Performance** to see the agent added to rankings
4. **Check Manual Review Queue** to see flagged content with categories
5. **Generate Agent Report** to view comprehensive performance analysis
6. **Test without agent name** to see manual review flagging system
7. **Verify Manual Review tab** is accessible even without ranked agents

## 🔍 Manual Review System

The system intelligently handles incomplete data and content safety violations without losing information:

### When Manual Review is Triggered:
- 🚨 **Content Safety Violation**: Inappropriate content detected by guardrails
- ❌ **No agent name identified** in the call
- ❌ **Quality scoring fails** (< 2 scores extracted)

### What Happens:
1. ✅ **Data is ALWAYS saved** - no information loss
2. 🏷️ **Flagged with** `needs_manual_review: true`
3. ⚠️ **Warning displayed** in UI with clear reasoning
4. 📊 **Excluded from agent rankings** (ensures accuracy)
5. 📋 **Available in Manual Review Queue** for human review
6. 🚨 **Content violations stored** with FLAGGED_ prefix

### Manual Review Scenarios:
| Scenario | Content Safe | Agent Name | Quality Score | Result |
|----------|-------------|-----------|---------------|--------|
| Full Success | ✅ Safe | ✅ Present | ✅ Complete | Saved + Ranked |
| Content Violation | 🚨 Flagged | N/A | ⏭️ Skipped | Saved for Review (FLAGGED_) |
| No Agent | ✅ Safe | ❌ Missing | ✅ Complete | Saved for Review |
| Scoring Failed | ✅ Safe | ✅ Present | ❌ Incomplete | Saved for Review |
| Both Missing | ✅ Safe | ❌ Missing | ❌ Incomplete | Saved for Review |

### Accessing Manual Review Queue:
1. Navigate to **Agent Performance** page
2. Click on **Manual Review Queue** tab
3. View all flagged items with:
   - 🚨 Content safety violations (with categories)
   - ⚠️ Missing agent name calls
   - ⚠️ Incomplete scoring calls
4. Review full transcripts and details
5. Tab badge shows count of items needing review

### Benefits:
- 🔒 **Zero Data Loss**: Every valid conversation is preserved
- 🛡️ **Content Safety**: Inappropriate content flagged for review
- 📊 **Accurate Rankings**: Only reliable scores affect agent performance
- 🔍 **Full Audit Trail**: All calls available for compliance and review
- 🎯 **Clear Attribution**: Manual review resolves ambiguous cases
- 👁️ **Easy Access**: Manual Review Queue always visible in Agent Performance page

## 🏗️ Architecture

The system uses **LangGraph** for multi-agent orchestration with 6 specialized agents:

```
Input (Text/Audio)
        ↓
    [Router] ─────→ Audio? → [Transcription Agent]
        ↓                            ↓
    Text? ──────────────────────────┘
        ↓
[Content Safety Agent] ← Check for inappropriate content (OpenAI Moderation API)
        ↓                 🆕 Routes to storage if flagged, continues if safe
        ├─────→ Flagged? → [Data Storage] (FLAGGED_ files)
        ↓
[Call Intake Agent] ← Extract metadata & parse conversation (LLM-based)
        ↓             ✨ Combined validation + extraction (50% cost reduction)
[Summarization Agent] ← Generate summary using GPT-4
        ↓
[Quality Scoring Agent] ← Evaluate quality with context-aware rubric
        ↓             ✨ Always runs (even without agent name for insights)
        ↓
[Data Storage Agent] ← ALWAYS SAVES DATA (zero loss!)
        ↓             ✨ Flags for manual review if incomplete or unsafe
    Results + Analytics + Manual Review Queue
```


## 💰 Cost Estimate

Approximate OpenAI API costs per call (after optimizations):
- **Whisper** (audio transcription): ~$0.006 per minute of audio
- **Moderation API** (content safety): ~$0.002 per call 🆕
- **GPT-4** (combined validation + metadata + summary): ~$0.02-0.04 per call ✨ (35% reduction!)
- **GPT-4** (quality scoring, always runs): ~$0.01-0.02 per call
- **Typical 5-minute call**: $0.04 - $0.16 ✨ (was $0.05-$0.20)



## 🔒 Security

- ✅ API keys stored in `.env` 
- ✅ Secure file handling with temporary storage for audio
- ✅ Local data storage (no external database required)

## 🐛 Troubleshooting

### "Import could not be resolved" errors
These are normal before installation. Run `pip install -r requirements.txt` to fix.

### "OPENAI_API_KEY not set"
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Restart the application

### "Module not found"
Ensure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```



Start by running: `streamlit run app.py`
