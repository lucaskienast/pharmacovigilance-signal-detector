"""
Audit Logger for Pharmacovigilance Signal Detector

Provides comprehensive audit trail functionality for regulatory compliance
and debugging of multi-agent pharmacovigilance analyses.

All significant events are logged with timestamps to JSON files for:
- Regulatory compliance (21 CFR Part 11 considerations)
- Debugging and troubleshooting
- Quality assurance review
- Educational demonstration of system behavior
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy


class AuditLogger:
    """
    Audit trail logger for pharmacovigilance analyses.
    
    Creates structured JSON logs for each analysis run, capturing:
    - Run metadata (drug, event, timestamps)
    - Agent executions and tool calls
    - Quality control evaluations
    - Feedback loop iterations
    - Final results and recommendations
    
    Usage:
        audit = AuditLogger()
        run_id = audit.start_run("METFORMIN", "LACTIC ACIDOSIS")
        audit.log_event("agent_executed", {"agent": "statistical", ...})
        audit.end_run(final_results)
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        """
        Initialize the audit logger.
        
        Args:
            log_dir: Directory to store audit log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.current_run: Optional[Dict[str, Any]] = None
        self.run_id: Optional[str] = None
        self.log_file: Optional[Path] = None
        
    def start_run(self, drug_name: str, adverse_event: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new audit run.
        
        Args:
            drug_name: Name of the drug being analyzed
            adverse_event: Adverse event being investigated
            metadata: Optional additional metadata
            
        Returns:
            Unique run ID for this analysis
        """
        self.run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()
        
        # Create filename with timestamp and drug name
        safe_drug = "".join(c if c.isalnum() else "_" for c in drug_name)[:20]
        filename = f"audit_{timestamp.strftime('%Y%m%d_%H%M%S')}_{safe_drug}.json"
        self.log_file = self.log_dir / filename
        
        self.current_run = {
            "run_id": self.run_id,
            "start_timestamp": timestamp.isoformat(),
            "end_timestamp": None,
            "drug_name": drug_name,
            "adverse_event": adverse_event,
            "metadata": metadata or {},
            "events": [],
            "iterations": [],
            "final_results": None,
            "status": "in_progress"
        }
        
        self.log_event("analysis_started", {
            "drug_name": drug_name,
            "adverse_event": adverse_event
        })
        
        # Save initial state
        self._save()
        
        return self.run_id
    
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None,
                  level: str = "info") -> None:
        """
        Log an event to the audit trail.
        
        Args:
            event_type: Type of event (e.g., "agent_executed", "quality_review")
            data: Event-specific data
            level: Log level ("info", "warning", "error")
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "level": level,
            "data": self._sanitize_for_json(data) if data else {}
        }
        
        self.current_run["events"].append(event)
        self._save()
    
    def log_iteration(self, iteration: int, quality_score: float,
                      issues: List[Dict], actions: List[Dict]) -> None:
        """
        Log a feedback loop iteration.
        
        Args:
            iteration: Iteration number (1-indexed)
            quality_score: Quality score after this iteration
            issues: Issues identified by QC
            actions: Actions taken (agent revisions)
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        iteration_record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "issues_found": len(issues),
            "issues": self._sanitize_for_json(issues),
            "actions_taken": self._sanitize_for_json(actions)
        }
        
        self.current_run["iterations"].append(iteration_record)
        
        self.log_event("iteration_completed", {
            "iteration": iteration,
            "quality_score": quality_score,
            "issues_count": len(issues),
            "actions_count": len(actions)
        })
    
    def log_agent_execution(self, agent_name: str, tool_calls: List[str],
                           success: bool, revision_number: int = 0,
                           feedback_context: Optional[str] = None) -> None:
        """
        Log an agent execution.
        
        Args:
            agent_name: Name of the agent
            tool_calls: List of tools called
            success: Whether execution succeeded
            revision_number: Revision number (0 for initial, 1+ for revisions)
            feedback_context: Feedback that triggered revision (if any)
        """
        self.log_event("agent_executed", {
            "agent": agent_name,
            "tool_calls": tool_calls,
            "success": success,
            "revision_number": revision_number,
            "is_revision": revision_number > 0,
            "feedback_context": feedback_context[:200] if feedback_context else None
        })
    
    def log_quality_review(self, scores: Dict[str, float], 
                          issues: List[Dict], 
                          approval_status: str,
                          target_agents: List[str]) -> None:
        """
        Log a quality control review.
        
        Args:
            scores: Quality scores (overall and dimensional)
            issues: Issues identified
            approval_status: "approved", "conditional", or "revision_required"
            target_agents: Agents that need revision
        """
        self.log_event("quality_review", {
            "scores": scores,
            "issues_count": len(issues),
            "issues_summary": [
                {"type": i.get("type"), "severity": i.get("severity"), 
                 "agent": i.get("responsible_agent")}
                for i in issues
            ],
            "approval_status": approval_status,
            "target_agents": target_agents
        })
    
    def log_literature_fetch(self, evidence: Dict[str, Any]) -> None:
        """
        Log literature evidence retrieval.
        
        Args:
            evidence: Literature evidence dictionary
        """
        self.log_event("literature_fetched", {
            "source": evidence.get("source"),
            "available": evidence.get("available"),
            "n_articles": evidence.get("n_articles"),
            "evidence_strength": evidence.get("evidence_strength"),
            "query": evidence.get("query")
        })
    
    def log_error(self, error_type: str, message: str, 
                  context: Optional[Dict] = None) -> None:
        """
        Log an error event.
        
        Args:
            error_type: Type of error
            message: Error message
            context: Additional context
        """
        self.log_event("error", {
            "error_type": error_type,
            "message": message,
            "context": context
        }, level="error")
    
    def end_run(self, final_results: Optional[Dict[str, Any]] = None,
                status: str = "completed") -> str:
        """
        End the current audit run.
        
        Args:
            final_results: Final analysis results
            status: Final status ("completed", "failed", "partial")
            
        Returns:
            Path to the saved audit log file
        """
        if self.current_run is None:
            raise RuntimeError("No active run to end.")
        
        self.current_run["end_timestamp"] = datetime.now().isoformat()
        self.current_run["status"] = status
        self.current_run["final_results"] = self._sanitize_for_json(final_results)
        
        # Calculate summary statistics
        self.current_run["summary"] = self._generate_summary()
        
        self.log_event("analysis_completed", {
            "status": status,
            "total_iterations": len(self.current_run["iterations"]),
            "final_quality_score": self._get_final_quality_score()
        })
        
        self._save()
        
        log_path = str(self.log_file)
        
        # Reset state
        self.current_run = None
        self.run_id = None
        
        return log_path
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the run."""
        events = self.current_run["events"]
        iterations = self.current_run["iterations"]
        
        agent_executions = [e for e in events if e["event_type"] == "agent_executed"]
        revisions = [e for e in agent_executions if e["data"].get("is_revision")]
        
        return {
            "total_events": len(events),
            "total_iterations": len(iterations),
            "total_agent_executions": len(agent_executions),
            "total_revisions": len(revisions),
            "agents_revised": list(set(e["data"]["agent"] for e in revisions)),
            "quality_score_progression": [
                {"iteration": i["iteration"], "score": i["quality_score"]}
                for i in iterations
            ],
            "errors_encountered": len([e for e in events if e["level"] == "error"])
        }
    
    def _get_final_quality_score(self) -> Optional[float]:
        """Get the final quality score from iterations."""
        if self.current_run["iterations"]:
            return self.current_run["iterations"][-1]["quality_score"]
        return None
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Sanitize an object for JSON serialization.
        
        Handles non-serializable types and truncates large strings.
        """
        if obj is None:
            return None
        
        if isinstance(obj, (str, int, float, bool)):
            # Truncate very long strings
            if isinstance(obj, str) and len(obj) > 5000:
                return obj[:5000] + "... [truncated]"
            return obj
        
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # For other types, convert to string
        try:
            return str(obj)
        except Exception:
            return "<non-serializable>"
    
    def _save(self) -> None:
        """Save current run to file."""
        if self.current_run and self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.current_run, f, indent=2, default=str)
    
    def get_run_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the current run (if active)."""
        if self.current_run is None:
            return None
        
        return {
            "run_id": self.run_id,
            "drug_name": self.current_run["drug_name"],
            "adverse_event": self.current_run["adverse_event"],
            "events_count": len(self.current_run["events"]),
            "iterations_count": len(self.current_run["iterations"]),
            "status": self.current_run["status"]
        }


def load_audit_log(filepath: str) -> Dict[str, Any]:
    """
    Load an audit log from a JSON file.
    
    Args:
        filepath: Path to the audit log file
        
    Returns:
        Audit log dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def list_audit_logs(log_dir: str = "audit_logs") -> List[Dict[str, str]]:
    """
    List all audit logs in the directory.
    
    Args:
        log_dir: Directory containing audit logs
        
    Returns:
        List of log file info dictionaries
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []
    
    logs = []
    for f in sorted(log_path.glob("audit_*.json"), reverse=True):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                logs.append({
                    "filename": f.name,
                    "filepath": str(f),
                    "run_id": data.get("run_id"),
                    "drug_name": data.get("drug_name"),
                    "adverse_event": data.get("adverse_event"),
                    "status": data.get("status"),
                    "timestamp": data.get("start_timestamp")
                })
        except Exception:
            continue
    
    return logs
