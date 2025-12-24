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
        A 16-character hex hash string for identity matching (64-bit).
        This provides sufficient collision resistance for typical job counts
        while remaining human-readable.
    """
    components = [kind, task]
    if seed is not None:
        components.append(str(seed))
    data = ":".join(components)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class HeartbeatWriter:
    """Atomic heartbeat writer for cluster job liveness tracking.
    
    Updates heartbeat.json every `interval` seconds with run metadata and progress.
    Uses atomic write (temp file + os.replace) to prevent partial reads.
    
    Uses a dedicated background thread with Event-based waiting for clean shutdown,
    avoiding the complexity and race conditions of chained Timer objects.
    
    Schema v1 required fields:
        - schema_version: 1
        - timestamp: ISO-8601 UTC
        - run_id: Unique run identifier
        - kind: 'train' or 'eval'
        - task: Task name
        - work_hash: Stable job identity for deduplication
        - progress.step: Current training step (non-decreasing)
        - job.scheduler: 'lsf'
        - job.job_id: LSF job ID (empty string "" if not running under LSF)
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
        # LSB_JOBID is the LSF batch job ID; empty string means not running under LSF
        self._job_id = os.environ.get("LSB_JOBID", "")
        
        # Progress tracking (protected by _lock)
        self._step = 0
        self._checkpoint_step: Optional[int] = None
        self._checkpoint_path: Optional[str] = None
        self._status = "running"
        
        # Threading: use Event-based thread instead of chained Timers
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def _snapshot_payload(self) -> dict:
        """Build the heartbeat JSON payload. Must be called with _lock held."""
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
    
    def _write_atomic(self, payload: dict) -> None:
        """Write heartbeat file atomically using temp + rename.
        
        Args:
            payload: Pre-built payload dict to write (snapshot taken under lock)
        """
        if not self.enabled:
            return
        
        try:
            # Ensure directory exists
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file
            with open(self._tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            
            # Atomic rename
            os.replace(self._tmp_path, self._heartbeat_path)
            
        except (OSError, IOError):
            # Silently ignore filesystem errors (non-critical for training)
            # Common cases: full disk, permission issues, network filesystem hiccups
            pass
    
    def _write_once(self) -> None:
        """Take snapshot under lock, then write atomically."""
        with self._lock:
            payload = self._snapshot_payload()
        self._write_atomic(payload)
    
    def _run_loop(self) -> None:
        """Background thread: write heartbeat periodically until stopped."""
        while not self._stop_event.wait(timeout=self.interval):
            self._write_once()
    
    def start(self) -> None:
        """Start the periodic heartbeat writer."""
        if not self.enabled:
            return
        
        with self._lock:
            if self._thread is not None:
                return  # Already started
            self._stop_event.clear()
        
        # Write initial heartbeat immediately
        self._write_once()
        
        # Start background thread for periodic updates
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the periodic heartbeat writer."""
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)  # Don't block forever
    
    def update_step(self, step: int) -> None:
        """Update the current training step.
        
        Args:
            step: Current training step (must be non-decreasing)
        """
        if not self.enabled:
            return
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
        if not self.enabled:
            return
        with self._lock:
            self._checkpoint_step = step
            self._checkpoint_path = path
    
    def update_status(self, status: str) -> None:
        """Update run status.
        
        Args:
            status: Status string ('running', 'stopping', etc.)
        """
        if not self.enabled:
            return
        with self._lock:
            self._status = status
    
    def write_now(self) -> None:
        """Write heartbeat immediately (useful before shutdown)."""
        self._write_once()
    
    def __enter__(self) -> "HeartbeatWriter":
        """Context manager entry: start the heartbeat writer."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: stop the heartbeat writer."""
        self.stop()
