"""
Atomic heartbeat writer for runctl liveness/progress tracking.

Creates heartbeat.json in the run's work directory with atomic updates (write to temp, then rename).
See issue #5: https://github.com/keyboardAnt/newt/issues/5
"""

import hashlib
import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def compute_work_hash(kind: str, task: str, seed: Optional[int] = None) -> str:
    """Compute stable job identity hash for dedupe + duplicate grouping.
    
    Args:
        kind: Job type ('train' or 'eval')
        task: Task name (e.g., 'walker-walk' or 'soup')
        seed: Optional seed value
    
    Returns:
        A short hash string for identity matching
    """
    components = [kind, task]
    if seed is not None:
        components.append(str(seed))
    data = ":".join(components)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


class HeartbeatWriter:
    """Atomic heartbeat writer for cluster job liveness tracking.
    
    Updates heartbeat.json every `interval` seconds with run metadata and progress.
    Uses atomic write (temp file + os.replace) to prevent partial reads.
    
    Schema v1 required fields:
        - schema_version: 1
        - timestamp: ISO-8601 UTC
        - run_id: Unique run identifier
        - kind: 'train' or 'eval'
        - task: Task name
        - work_hash: Stable job identity for deduplication
        - progress.step: Current training step (non-decreasing)
        - job.scheduler: 'lsf'
        - job.job_id: LSF job ID
        - host.hostname: Machine hostname
        - host.pid: Process ID
    
    TTL semantics:
        - alive: age <= 120s
        - maybe-stale: 120s < age <= 600s
    """
    
    SCHEMA_VERSION = 1
    DEFAULT_INTERVAL = 30.0  # seconds
    
    def __init__(
        self,
        work_dir: str,
        run_id: str,
        kind: str,
        task: str,
        seed: Optional[int] = None,
        interval: float = DEFAULT_INTERVAL,
        enabled: bool = True,
    ):
        """Initialize the heartbeat writer.
        
        Args:
            work_dir: Directory to write heartbeat.json
            run_id: Unique run identifier
            kind: Job type ('train' or 'eval')
            task: Task name
            seed: Optional seed value
            interval: Update interval in seconds (default: 30)
            enabled: Whether heartbeat is enabled (disabled for rank > 0)
        """
        self.work_dir = Path(work_dir)
        self.run_id = run_id
        self.kind = kind
        self.task = task
        self.seed = seed
        self.interval = interval
        self.enabled = enabled
        
        # Compute stable identity hash
        self.work_hash = compute_work_hash(kind, task, seed)
        
        # File paths
        self._heartbeat_path = self.work_dir / "heartbeat.json"
        self._tmp_path = self.work_dir / "heartbeat.json.tmp"
        
        # Host and job metadata (computed once)
        self._hostname = socket.gethostname()
        self._pid = os.getpid()
        self._job_id = os.environ.get("LSB_JOBID", "")
        
        # Progress tracking
        self._step = 0
        self._checkpoint_step: Optional[int] = None
        self._checkpoint_path: Optional[str] = None
        self._status = "running"
        
        # Timer for periodic updates
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._stopped = False
    
    def _build_payload(self) -> dict:
        """Build the heartbeat JSON payload."""
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "kind": self.kind,
            "task": self.task,
            "work_hash": self.work_hash,
            "progress": {
                "step": self._step,
            },
            "job": {
                "scheduler": "lsf",
                "job_id": self._job_id,
            },
            "host": {
                "hostname": self._hostname,
                "pid": self._pid,
            },
        }
        
        # Optional fields
        if self.seed is not None:
            payload["seed"] = self.seed
        
        if self._checkpoint_step is not None:
            payload["checkpoint"] = {
                "step": self._checkpoint_step,
            }
            if self._checkpoint_path is not None:
                payload["checkpoint"]["path"] = self._checkpoint_path
        
        # Status is always included (initialized to 'running')
        payload["status"] = self._status
        
        return payload
    
    def _write_atomic(self) -> None:
        """Write heartbeat file atomically using temp + rename."""
        if not self.enabled:
            return
        
        try:
            # Ensure directory exists
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            payload = self._build_payload()
            
            # Write to temp file
            with open(self._tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            
            # Atomic rename
            os.replace(self._tmp_path, self._heartbeat_path)
            
        except (OSError, IOError):
            # Silently ignore filesystem errors (non-critical for training)
            # Common cases: full disk, permission issues, network filesystem hiccups
            pass
    
    def _schedule_next(self) -> None:
        """Schedule the next heartbeat write."""
        if self._stopped or not self.enabled:
            return
        
        self._timer = threading.Timer(self.interval, self._tick)
        self._timer.daemon = True
        self._timer.start()
    
    def _tick(self) -> None:
        """Timer callback: write heartbeat and schedule next."""
        with self._lock:
            if self._stopped:
                return
            self._write_atomic()
            self._schedule_next()
    
    def start(self) -> None:
        """Start the periodic heartbeat writer."""
        if not self.enabled:
            return
        
        with self._lock:
            if self._stopped:
                return
            # Write initial heartbeat immediately
            self._write_atomic()
            # Schedule periodic updates
            self._schedule_next()
    
    def stop(self) -> None:
        """Stop the periodic heartbeat writer."""
        with self._lock:
            self._stopped = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
    
    def update_step(self, step: int) -> None:
        """Update the current training step.
        
        Args:
            step: Current training step (must be non-decreasing)
        """
        with self._lock:
            # Ensure step is non-decreasing
            if step >= self._step:
                self._step = step
    
    def update_checkpoint(self, step: int, path: Optional[str] = None) -> None:
        """Update checkpoint information.
        
        Args:
            step: Checkpoint step
            path: Optional checkpoint file path
        """
        with self._lock:
            self._checkpoint_step = step
            self._checkpoint_path = path
    
    def update_status(self, status: str) -> None:
        """Update run status.
        
        Args:
            status: Status string ('running', 'stopping', etc.)
        """
        with self._lock:
            self._status = status
    
    def write_now(self) -> None:
        """Write heartbeat immediately (useful before shutdown)."""
        with self._lock:
            self._write_atomic()
    
    def __enter__(self) -> "HeartbeatWriter":
        """Context manager entry: start the heartbeat writer."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: stop the heartbeat writer."""
        self.stop()
