"""
Data Storage Agent - Stores call data and quality scores for analytics.
"""
import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from utils.models import CallData, CallSummary, QualityScore


class DataStorageAgent:
    """
    Agent responsible for persisting call data, summaries, and quality scores
    to various data stores for analytics and agent evaluation.
    """
    
    def __init__(self, storage_dir: str = "data_storage_call_center"):
        """
        Initialize the Data Storage Agent.
        
        Args:
            storage_dir: Directory where data will be stored
        """
        self.name = "Data Storage Agent"
        self.storage_dir = Path(storage_dir)
        
        # Create storage directories
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / "calls").mkdir(exist_ok=True)
        (self.storage_dir / "scores").mkdir(exist_ok=True)
        (self.storage_dir / "reports").mkdir(exist_ok=True)
        
        # File paths
        self.calls_db = self.storage_dir / "calls_database.json"
        self.scores_csv = self.storage_dir / "quality_scores.csv"
        self.agent_performance_csv = self.storage_dir / "agent_performance.csv"
        
        # Initialize storage files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage files if they don't exist."""
        # Initialize JSON database
        if not self.calls_db.exists():
            with open(self.calls_db, 'w') as f:
                json.dump({"calls": []}, f, indent=2)
        
        # Initialize CSV files with headers
        if not self.scores_csv.exists():
            self._create_scores_csv()
        
        if not self.agent_performance_csv.exists():
            self._create_agent_performance_csv()
    
    def _create_scores_csv(self):
        """Create the quality scores CSV file with headers."""
        headers = [
            'timestamp', 'call_id', 'agent_name', 'caller_name',
            'call_duration', 'overall_score', 'tone_score', 
            'professionalism_score', 'resolution_score', 'response_score',
            'feedback', 'strengths', 'areas_for_improvement'
        ]
        with open(self.scores_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def _create_agent_performance_csv(self):
        """Create the agent performance summary CSV file."""
        headers = [
            'agent_name', 'total_calls', 'average_overall_score',
            'average_tone_score', 'average_professionalism_score',
            'average_resolution_score', 'average_response_score',
            'last_updated'
        ]
        with open(self.agent_performance_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and store call data, summary, and quality scores.
        Stores ALL calls (even without agent name) for manual review.
        Also stores flagged content from content safety checks.
        Only updates agent performance metrics when agent is identified.
        
        Args:
            state: Current agent state containing all processed data
            
        Returns:
            Updated state with storage confirmation
        """
        try:
            call_data = state.get("call_data")
            summary = state.get("summary")
            quality_score = state.get("quality_score")
            needs_manual_review = state.get("needs_manual_review", False)
            content_safety_passed = state.get("content_safety_passed", True)
            content_safety_details = state.get("content_safety_details", {})
            
            # Handle content safety flagged items (may not have call_data yet)
            if not content_safety_passed:
                return self._store_flagged_content(state)
            
            if not call_data:
                return {
                    **state,
                    "processing_steps": ["Data Storage: Skipped - Missing call data"]
                }
                return {
                    **state,
                    "processing_steps": ["Data Storage: Skipped - Missing call data"]
                }
            
            # Generate unique call ID if not present
            if not call_data.metadata.call_id:
                call_data.metadata.call_id = self._generate_call_id()
            
            # Check if agent name is missing (flag for manual review)
            has_agent_name = (call_data.metadata.agent_name and 
                            call_data.metadata.agent_name.strip())
            
            if not has_agent_name:
                # Mark for manual review if no agent name
                needs_manual_review = True
            
            # Store complete call record (ALWAYS - even without agent name)
            self._store_call_record(call_data, summary, quality_score, needs_manual_review)
            
            # Only store quality score and update analytics if we have agent name and valid score
            if has_agent_name and quality_score:
                self._store_quality_score(call_data, quality_score)
                # Update agent performance analytics (only with valid scores)
                self._update_agent_performance(call_data, quality_score)
            
            # Build success message
            if not has_agent_name:
                message = f"📋 Data Storage: Call {call_data.metadata.call_id} saved for manual review (no agent identified)"
            elif needs_manual_review:
                message = f"Data Storage: Call {call_data.metadata.call_id} saved for manual review (scoring incomplete)"
            else:
                message = f"Data Storage: Call {call_data.metadata.call_id} stored successfully"
            
            # Update state - return new list for operator.add
            state["storage_path"] = str(self.storage_dir)
            
            return {
                **state,
                "processing_steps": [message]
            }
            
        except Exception as e:
            state["error"] = f"Data Storage Agent error: {str(e)}"
            return state
    
    def _generate_call_id(self) -> str:
        """Generate a unique call ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"CALL_{timestamp}"
    
    def _store_flagged_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store content that failed content safety checks for manual review.
        
        Args:
            state: Current state with transcript and content safety details
            
        Returns:
            Updated state with storage confirmation
        """
        transcript = state.get("transcript", "")
        content_safety_details = state.get("content_safety_details", {})
        input_type = state.get("input_type", "unknown")
        
        # Generate call ID for flagged content
        call_id = f"FLAGGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create flagged content record
        flagged_record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "status": "FLAGGED_FOR_REVIEW",
            "needs_manual_review": True,
            "content_safety_passed": False,
            "flagged_categories": content_safety_details.get("flagged_categories", []),
            "input_type": input_type,
            "transcript": transcript,
            "metadata": {
                "agent_name": None,
                "caller_name": None,
                "call_duration": None,
                "date_time": datetime.now().isoformat()
            }
        }
        
        # Save to individual file
        flagged_file = self.storage_dir / "calls" / f"{call_id}.json"
        with open(flagged_file, 'w', encoding='utf-8') as f:
            json.dump(flagged_record, f, indent=2, ensure_ascii=False)
        
        # Add to main database
        with open(self.calls_db, 'r', encoding='utf-8') as f:
            db = json.load(f)
        
        db["calls"].append({
            "call_id": call_id,
            "timestamp": flagged_record["timestamp"],
            "agent_name": None,
            "overall_score": None,
            "needs_manual_review": True,
            "status": "FLAGGED_FOR_REVIEW",
            "flagged_categories": content_safety_details.get("flagged_categories", [])
        })
        
        with open(self.calls_db, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        
        # Update state
        flagged_categories = ', '.join(content_safety_details.get("flagged_categories", []))
        
        return {
            **state,
            "storage_path": str(self.storage_dir),
            "processing_steps": [
                f"🚨 Data Storage: Flagged content stored as {call_id}",
                f"⚠️  Reason: {flagged_categories}",
                f"🔍 Status: Awaiting manual review"
            ]
        }
    
    def _store_call_record(
        self, 
        call_data: CallData, 
        summary: Optional[CallSummary], 
        quality_score: Optional[QualityScore],
        needs_manual_review: bool = False
    ):
        """
        Store complete call record as JSON.
        
        Args:
            call_data: Call data with metadata and conversation
            summary: Call summary
            quality_score: Quality evaluation (can be None if scoring failed)
            needs_manual_review: Flag indicating call needs manual review
        """
        # Create record
        record = {
            "call_id": call_data.metadata.call_id,
            "timestamp": datetime.now().isoformat(),
            "needs_manual_review": needs_manual_review,
            "metadata": {
                "agent_name": call_data.metadata.agent_name,
                "caller_name": call_data.metadata.caller_name,
                "call_duration": call_data.metadata.call_duration,
                "date_time": call_data.metadata.date_time
            },
            "conversation": call_data.conversation,
            "conversation_turns": [
                {"speaker": turn.speaker, "text": turn.text}
                for turn in call_data.conversation_turns
            ] if call_data.conversation_turns else [],
            "summary": {
                "brief_summary": summary.brief_summary,
                "key_points": summary.key_points,
                "customer_issue": summary.customer_issue,
                "resolution": summary.resolution,
                "action_items": summary.action_items
            } if summary else None,
            "quality_score": {
                "overall_score": quality_score.overall_score,
                "tone_score": quality_score.tone_score,
                "professionalism_score": quality_score.professionalism_score,
                "resolution_score": quality_score.resolution_score,
                "response_time_score": quality_score.response_time_score,
                "feedback": quality_score.feedback,
                "strengths": quality_score.strengths,
                "areas_for_improvement": quality_score.areas_for_improvement
            } if quality_score else None
        }
        
        # Save to individual file
        call_file = self.storage_dir / "calls" / f"{call_data.metadata.call_id}.json"
        with open(call_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # Add to main database
        with open(self.calls_db, 'r', encoding='utf-8') as f:
            db = json.load(f)
        
        db["calls"].append({
            "call_id": call_data.metadata.call_id,
            "timestamp": record["timestamp"],
            "agent_name": call_data.metadata.agent_name,
            "overall_score": quality_score.overall_score if quality_score else None,
            "needs_manual_review": needs_manual_review
        })
        
        with open(self.calls_db, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    
    def _store_quality_score(self, call_data: CallData, quality_score: QualityScore):
        """
        Store quality score to CSV for easy analysis.
        
        Args:
            call_data: Call data with metadata
            quality_score: Quality evaluation
        """
        row = {
            'timestamp': datetime.now().isoformat(),
            'call_id': call_data.metadata.call_id,
            'agent_name': call_data.metadata.agent_name or 'Unknown',
            'caller_name': call_data.metadata.caller_name or 'Unknown',
            'call_duration': call_data.metadata.call_duration or 'N/A',
            'overall_score': round(quality_score.overall_score, 2),
            'tone_score': round(quality_score.tone_score, 2),
            'professionalism_score': round(quality_score.professionalism_score, 2),
            'resolution_score': round(quality_score.resolution_score, 2),
            'response_score': round(quality_score.response_time_score, 2),
            'feedback': quality_score.feedback[:200] if quality_score.feedback else '',  # Truncate
            'strengths': ' | '.join(quality_score.strengths) if quality_score.strengths else '',
            'areas_for_improvement': ' | '.join(quality_score.areas_for_improvement) 
                if quality_score.areas_for_improvement else ''
        }
        
        # Append to CSV
        with open(self.scores_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def _update_agent_performance(self, call_data: CallData, quality_score: QualityScore):
        """
        Update agent performance metrics.
        
        Args:
            call_data: Call data with metadata
            quality_score: Quality evaluation
        """
        agent_name = call_data.metadata.agent_name or 'Unknown'
        
        # Read existing performance data
        if self.agent_performance_csv.exists() and os.path.getsize(self.agent_performance_csv) > 0:
            try:
                df = pd.read_csv(self.agent_performance_csv)
            except:
                # Create empty DataFrame with correct schema
                df = pd.DataFrame(columns=[
                    'agent_name', 'total_calls', 'average_overall_score',
                    'average_tone_score', 'average_professionalism_score',
                    'average_resolution_score', 'average_response_score',
                    'last_updated'
                ])
        else:
            # Create empty DataFrame with correct schema
            df = pd.DataFrame(columns=[
                'agent_name', 'total_calls', 'average_overall_score',
                'average_tone_score', 'average_professionalism_score',
                'average_resolution_score', 'average_response_score',
                'last_updated'
            ])
        
        # Calculate new metrics
        if agent_name in df['agent_name'].values if not df.empty else False:
            # Update existing agent
            mask = df['agent_name'] == agent_name
            current_calls = df.loc[mask, 'total_calls'].values[0]
            new_calls = current_calls + 1
            
            # Calculate running averages
            df.loc[mask, 'average_overall_score'] = (
                (df.loc[mask, 'average_overall_score'] * current_calls + quality_score.overall_score) 
                / new_calls
            )
            df.loc[mask, 'average_tone_score'] = (
                (df.loc[mask, 'average_tone_score'] * current_calls + quality_score.tone_score) 
                / new_calls
            )
            df.loc[mask, 'average_professionalism_score'] = (
                (df.loc[mask, 'average_professionalism_score'] * current_calls + 
                 quality_score.professionalism_score) / new_calls
            )
            df.loc[mask, 'average_resolution_score'] = (
                (df.loc[mask, 'average_resolution_score'] * current_calls + 
                 quality_score.resolution_score) / new_calls
            )
            df.loc[mask, 'average_response_score'] = (
                (df.loc[mask, 'average_response_score'] * current_calls + 
                 quality_score.response_time_score) / new_calls
            )
            df.loc[mask, 'total_calls'] = new_calls
            df.loc[mask, 'last_updated'] = datetime.now().isoformat()
        else:
            # Add new agent - using loc to avoid FutureWarning
            new_data = {
                'agent_name': agent_name,
                'total_calls': 1,
                'average_overall_score': quality_score.overall_score,
                'average_tone_score': quality_score.tone_score,
                'average_professionalism_score': quality_score.professionalism_score,
                'average_resolution_score': quality_score.resolution_score,
                'average_response_score': quality_score.response_time_score,
                'last_updated': datetime.now().isoformat()
            }
            # Use loc to add new row (more efficient than concat)
            df.loc[len(df)] = new_data
        
        # Save updated performance data
        df.to_csv(self.agent_performance_csv, index=False)
    
    def get_agent_performance(self, agent_name: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve agent performance metrics.
        
        Args:
            agent_name: Specific agent name, or None for all agents
            
        Returns:
            DataFrame with agent performance data
        """
        if not self.agent_performance_csv.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.agent_performance_csv)
        
        if agent_name:
            df = df[df['agent_name'] == agent_name]
        
        return df.sort_values('average_overall_score', ascending=False)
    
    def get_quality_scores(
        self, 
        agent_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve quality scores with optional filtering.
        
        Args:
            agent_name: Filter by agent name
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            DataFrame with quality scores
        """
        if not self.scores_csv.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.scores_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply filters
        if agent_name:
            df = df[df['agent_name'] == agent_name]
        
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        return df.sort_values('timestamp', ascending=False)
    
    def generate_agent_report(self, agent_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dictionary with performance metrics and insights
        """
        # Get all scores for this agent
        scores_df = self.get_quality_scores(agent_name=agent_name)
        
        if scores_df.empty:
            return {"error": f"No data found for agent: {agent_name}"}
        
        # Calculate statistics
        report = {
            "agent_name": agent_name,
            "total_calls": len(scores_df),
            "date_range": {
                "start": scores_df['timestamp'].min().isoformat(),
                "end": scores_df['timestamp'].max().isoformat()
            },
            "scores": {
                "overall": {
                    "average": round(scores_df['overall_score'].mean(), 2),
                    "median": round(scores_df['overall_score'].median(), 2),
                    "min": round(scores_df['overall_score'].min(), 2),
                    "max": round(scores_df['overall_score'].max(), 2),
                    "std_dev": round(scores_df['overall_score'].std(), 2)
                },
                "tone": {
                    "average": round(scores_df['tone_score'].mean(), 2),
                    "trend": self._calculate_trend(scores_df['tone_score'])
                },
                "professionalism": {
                    "average": round(scores_df['professionalism_score'].mean(), 2),
                    "trend": self._calculate_trend(scores_df['professionalism_score'])
                },
                "resolution": {
                    "average": round(scores_df['resolution_score'].mean(), 2),
                    "trend": self._calculate_trend(scores_df['resolution_score'])
                },
                "response": {
                    "average": round(scores_df['response_score'].mean(), 2),
                    "trend": self._calculate_trend(scores_df['response_score'])
                }
            },
            "performance_rating": self._get_performance_rating(
                scores_df['overall_score'].mean()
            ),
            "recent_calls": scores_df.head(5)[
                ['timestamp', 'call_id', 'overall_score']
            ].to_dict('records')
        }
        
        # Save report
        report_file = self.storage_dir / "reports" / f"{agent_name}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _calculate_trend(self, scores: pd.Series) -> str:
        """Calculate trend (improving, declining, stable)."""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid = len(scores) // 2
        first_half = scores.iloc[:mid].mean()
        second_half = scores.iloc[mid:].mean()
        
        diff = second_half - first_half
        
        if diff > 0.5:
            return "improving"
        elif diff < -0.5:
            return "declining"
        else:
            return "stable"
    
    def _get_performance_rating(self, avg_score: float) -> str:
        """Get performance rating based on average score."""
        if avg_score >= 9.0:
            return "Outstanding"
        elif avg_score >= 8.0:
            return "Excellent"
        elif avg_score >= 7.0:
            return "Good"
        elif avg_score >= 6.0:
            return "Satisfactory"
        elif avg_score >= 5.0:
            return "Needs Improvement"
        else:
            return "Unsatisfactory"
    
    def get_manual_review_calls(self) -> List[Dict[str, Any]]:
        """
        Get all calls that need manual review.
        
        Returns:
            List of call records flagged for manual review
        """
        if not self.calls_db.exists():
            return []
        
        with open(self.calls_db, 'r', encoding='utf-8') as f:
            db = json.load(f)
        
        # Filter calls that need manual review
        manual_review_calls = [
            call for call in db.get("calls", [])
            if call.get("needs_manual_review", False)
        ]
        
        # Load full details for each call
        detailed_calls = []
        for call_summary in manual_review_calls:
            call_id = call_summary.get("call_id")
            call_file = self.storage_dir / "calls" / f"{call_id}.json"
            
            if call_file.exists():
                with open(call_file, 'r', encoding='utf-8') as f:
                    full_call = json.load(f)
                    detailed_calls.append(full_call)
        
        # Sort by timestamp (most recent first)
        detailed_calls.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return detailed_calls
