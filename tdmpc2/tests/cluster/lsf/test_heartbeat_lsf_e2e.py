#!/usr/bin/env python3
"""
End-to-end test for the heartbeat system on LSF cluster.

This script acts as a "watcher" that validates the heartbeat contract from issue #5:
- File appears within 60s at logs/<run_id>/heartbeat.json
- Atomic writes (no partial JSON / decode errors)
- Updates every ~30s
- progress.step is non-decreasing and increases during training
- Required schema fields are present
- Status becomes "stopping" on shutdown

Usage:
    # As watcher for a running training job (discovers run_id automatically):
    python test_heartbeat_e2e.py --logs-dir /path/to/logs --timeout 180

    # Watch a specific run:
    python test_heartbeat_e2e.py --run-id 20241224_120000_test --logs-dir /path/to/logs

    # Run the full e2e test (launches training + watcher):
    python test_heartbeat_e2e.py --full-test --logs-dir /path/to/logs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any


def parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, handling both Z and +00:00 UTC formats.
    
    Args:
        ts: Timestamp string (e.g., "2025-12-24T01:52:23.114Z" or with +00:00)
    
    Returns:
        datetime object (timezone-aware)
    """
    # Normalize Z suffix to +00:00 for fromisoformat compatibility
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


# Contract thresholds from issue #5
FILE_APPEAR_TIMEOUT = 60.0  # seconds
HEARTBEAT_INTERVAL = 30.0  # expected interval
CADENCE_MIN = 15.0  # allow some variance
CADENCE_MAX = 90.0  # allow scheduler delays
POLL_INTERVAL = 0.5  # how often we check the file
SHUTDOWN_TIMEOUT = 60.0  # time to wait for "stopping" status after training exits

# Required schema v1 fields (nested paths use dots)
REQUIRED_FIELDS = [
    "schema_version",
    "timestamp",
    "run_id",
    "kind",
    "task",
    "work_hash",
    "progress.step",
    "job.scheduler",
    "job.job_id",
    "host.hostname",
    "host.pid",
    "status",
]


class HeartbeatWatcher:
    """Watches and validates heartbeat.json for a training run."""

    def __init__(self, logs_dir: Path, run_id: Optional[str] = None, verbose: bool = True):
        self.logs_dir = Path(logs_dir)
        self.run_id = run_id
        self.verbose = verbose

        # Collected data
        self.heartbeats: List[Dict[str, Any]] = []
        self.timestamps: List[datetime] = []
        self.steps: List[int] = []
        self.decode_errors: int = 0
        self.first_seen_at: Optional[float] = None

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[watcher] {msg}", flush=True)

    def discover_run_id(self, start_time: float, timeout: float = FILE_APPEAR_TIMEOUT) -> Optional[str]:
        """Discover the run_id by finding the newest logs directory created after start_time."""
        deadline = time.time() + timeout
        self.log(f"Discovering run_id (waiting up to {timeout}s for new logs dir)...")

        while time.time() < deadline:
            if not self.logs_dir.exists():
                time.sleep(POLL_INTERVAL)
                continue

            # Find directories created after start_time
            candidates = []
            for d in self.logs_dir.iterdir():
                if d.is_dir():
                    try:
                        ctime = d.stat().st_ctime
                        if ctime >= start_time:
                            candidates.append((ctime, d.name))
                    except OSError:
                        pass

            if candidates:
                candidates.sort(reverse=True)
                run_id = candidates[0][1]
                self.log(f"Discovered run_id: {run_id}")
                return run_id

            time.sleep(POLL_INTERVAL)

        return None

    def get_heartbeat_path(self) -> Path:
        assert self.run_id is not None
        return self.logs_dir / self.run_id / "heartbeat.json"

    def wait_for_file(self, timeout: float = FILE_APPEAR_TIMEOUT) -> bool:
        """Wait for heartbeat.json to appear. Returns True if found."""
        hb_path = self.get_heartbeat_path()
        self.log(f"Waiting for {hb_path} (timeout: {timeout}s)...")

        start = time.time()
        while time.time() - start < timeout:
            if hb_path.exists():
                self.first_seen_at = time.time() - start
                self.log(f"File appeared after {self.first_seen_at:.1f}s")
                return True
            time.sleep(POLL_INTERVAL)

        self.log(f"ERROR: File did not appear within {timeout}s")
        return False

    def read_heartbeat(self) -> Optional[Dict[str, Any]]:
        """Try to read and parse heartbeat.json. Returns None on error."""
        hb_path = self.get_heartbeat_path()
        try:
            with open(hb_path, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            self.decode_errors += 1
            self.log(f"WARNING: JSON decode error (count={self.decode_errors}): {e}")
            return None
        except (OSError, IOError) as e:
            # File might be in the middle of atomic rename; retry
            return None

    def get_nested(self, data: dict, path: str) -> Any:
        """Get nested value using dot notation (e.g., 'progress.step')."""
        parts = path.split(".")
        val = data
        for p in parts:
            if not isinstance(val, dict) or p not in val:
                return None
            val = val[p]
        return val

    def validate_schema(self, data: dict) -> List[str]:
        """Validate required fields. Returns list of missing/invalid fields."""
        errors = []
        for field in REQUIRED_FIELDS:
            val = self.get_nested(data, field)
            if val is None:
                errors.append(f"missing: {field}")

        # Type checks
        if data.get("schema_version") != 1:
            errors.append(f"schema_version should be 1, got {data.get('schema_version')}")

        step = self.get_nested(data, "progress.step")
        if step is not None and not isinstance(step, int):
            errors.append(f"progress.step should be int, got {type(step).__name__}")

        pid = self.get_nested(data, "host.pid")
        if pid is not None and not isinstance(pid, int):
            errors.append(f"host.pid should be int, got {type(pid).__name__}")

        scheduler = self.get_nested(data, "job.scheduler")
        if scheduler != "lsf":
            errors.append(f"job.scheduler should be 'lsf', got {scheduler!r}")

        return errors

    def validate_lsf_job_id(self, data: dict) -> Optional[str]:
        """Validate job_id matches LSB_JOBID env var (if set)."""
        expected = os.environ.get("LSB_JOBID", "")
        actual = self.get_nested(data, "job.job_id")
        if actual != expected:
            return f"job.job_id mismatch: expected {expected!r}, got {actual!r}"
        return None

    def poll_until_done(
        self,
        training_process: Optional[subprocess.Popen] = None,
        max_duration: float = 300.0,
    ) -> None:
        """Poll heartbeat.json until training completes or timeout."""
        self.log(f"Polling heartbeat (max {max_duration}s)...")
        start = time.time()

        while time.time() - start < max_duration:
            # Check if training exited
            if training_process is not None:
                ret = training_process.poll()
                if ret is not None:
                    self.log(f"Training process exited with code {ret}")
                    break

            data = self.read_heartbeat()
            if data is not None:
                self.heartbeats.append(data)

                # Parse timestamp (handles both Z and +00:00 formats)
                try:
                    ts = parse_timestamp(data["timestamp"])
                    self.timestamps.append(ts)
                except (KeyError, ValueError):
                    pass

                # Parse step
                step = self.get_nested(data, "progress.step")
                if isinstance(step, int):
                    self.steps.append(step)

            time.sleep(POLL_INTERVAL)

        # After training exits, wait for "stopping" status
        if training_process is not None:
            self.log(f"Waiting up to {SHUTDOWN_TIMEOUT}s for 'stopping' status...")
            shutdown_start = time.time()
            while time.time() - shutdown_start < SHUTDOWN_TIMEOUT:
                data = self.read_heartbeat()
                if data is not None:
                    self.heartbeats.append(data)
                    if data.get("status") == "stopping":
                        self.log("Observed 'stopping' status")
                        break
                time.sleep(POLL_INTERVAL)

    def compute_cadences(self) -> List[float]:
        """Compute time deltas between consecutive heartbeats."""
        if len(self.timestamps) < 2:
            return []
        cadences = []
        for i in range(1, len(self.timestamps)):
            delta = (self.timestamps[i] - self.timestamps[i - 1]).total_seconds()
            cadences.append(delta)
        return cadences

    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report of the test results."""
        cadences = self.compute_cadences()

        report = {
            "run_id": self.run_id,
            "file_appeared_after_s": self.first_seen_at,
            "total_heartbeats": len(self.heartbeats),
            "unique_timestamps": len(set(str(t) for t in self.timestamps)),
            "decode_errors": self.decode_errors,
            "steps": {
                "count": len(self.steps),
                "min": min(self.steps) if self.steps else None,
                "max": max(self.steps) if self.steps else None,
                "increased": len(set(self.steps)) > 1 if self.steps else False,
            },
            "cadences": {
                "count": len(cadences),
                "min": min(cadences) if cadences else None,
                "max": max(cadences) if cadences else None,
                "avg": sum(cadences) / len(cadences) if cadences else None,
            },
        }

        # Check last status
        if self.heartbeats:
            report["final_status"] = self.heartbeats[-1].get("status")

        return report

    def validate_contract(self) -> List[str]:
        """Validate the heartbeat contract. Returns list of failures."""
        failures = []

        # 1. File appeared within timeout
        if self.first_seen_at is None:
            failures.append(f"File never appeared")
        elif self.first_seen_at > FILE_APPEAR_TIMEOUT:
            failures.append(f"File appeared too late: {self.first_seen_at:.1f}s > {FILE_APPEAR_TIMEOUT}s")

        # 2. No decode errors (atomicity)
        if self.decode_errors > 0:
            failures.append(f"JSON decode errors: {self.decode_errors} (atomicity violation)")

        # 3. Got at least 2 heartbeat updates
        unique_ts = len(set(str(t) for t in self.timestamps))
        if unique_ts < 2:
            failures.append(f"Too few heartbeat updates: {unique_ts} < 2")

        # 4. Cadence within expected range
        cadences = self.compute_cadences()
        for i, c in enumerate(cadences):
            if c < CADENCE_MIN:
                failures.append(f"Cadence[{i}] too fast: {c:.1f}s < {CADENCE_MIN}s")
            if c > CADENCE_MAX:
                failures.append(f"Cadence[{i}] too slow: {c:.1f}s > {CADENCE_MAX}s")

        # 5. Steps are non-decreasing
        for i in range(1, len(self.steps)):
            if self.steps[i] < self.steps[i - 1]:
                failures.append(f"Step decreased: {self.steps[i-1]} -> {self.steps[i]}")

        # 6. Steps increased at least once
        if len(set(self.steps)) <= 1 and len(self.steps) > 1:
            failures.append(f"Step never increased (all values: {self.steps[0] if self.steps else 'N/A'})")

        # 7. Schema validation on first heartbeat
        if self.heartbeats:
            schema_errors = self.validate_schema(self.heartbeats[0])
            for e in schema_errors:
                failures.append(f"Schema: {e}")

            lsf_error = self.validate_lsf_job_id(self.heartbeats[0])
            if lsf_error:
                failures.append(lsf_error)

        # 8. Final status should be "stopping"
        if self.heartbeats:
            final_status = self.heartbeats[-1].get("status")
            if final_status != "stopping":
                failures.append(f"Final status should be 'stopping', got {final_status!r}")

        return failures


def run_watcher_only(args: argparse.Namespace) -> int:
    """Run just the watcher against an existing or discovered run."""
    # In watcher-only mode, logs_dir must exist (we're watching an existing run)
    if not args.logs_dir.exists():
        print(f"ERROR: Logs directory does not exist: {args.logs_dir}", file=sys.stderr)
        return 1

    watcher = HeartbeatWatcher(logs_dir=args.logs_dir, run_id=args.run_id)

    # Discover run_id if not provided
    if watcher.run_id is None:
        start_time = time.time()
        watcher.run_id = watcher.discover_run_id(start_time, timeout=args.timeout)
        if watcher.run_id is None:
            print("ERROR: Could not discover run_id", file=sys.stderr)
            return 1

    # Wait for file
    if not watcher.wait_for_file():
        return 1

    # Poll until timeout
    watcher.poll_until_done(max_duration=args.timeout)

    # Report
    report = watcher.generate_report()
    print("\n" + "=" * 60)
    print("HEARTBEAT E2E TEST REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))

    # Validate
    failures = watcher.validate_contract()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("\nAll contract checks PASSED")
        return 0


def run_full_test(args: argparse.Namespace) -> int:
    """Launch training + watcher together."""
    watcher = HeartbeatWatcher(logs_dir=args.logs_dir)

    # Build training command
    train_cmd = [
        sys.executable,
        "train.py",
        f"task={args.task}",
        f"steps={args.steps}",
        f"seed={args.seed}",
        "model_size=S",  # smallest model
        "num_envs=1",
        "enable_wandb=False",
        "save_video=False",
        "compile=False",  # skip compilation overhead
        f"exp_name=heartbeat_e2e_test",
    ]

    watcher.log(f"Launching training: {' '.join(train_cmd)}")
    start_time = time.time()

    # Start training in background
    train_proc = subprocess.Popen(
        train_cmd,
        cwd=args.logs_dir.parent,  # tdmpc2 directory
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Discover run_id
        watcher.run_id = watcher.discover_run_id(start_time)
        if watcher.run_id is None:
            watcher.log("ERROR: Could not discover run_id")
            train_proc.terminate()
            return 1

        # Wait for heartbeat file
        if not watcher.wait_for_file():
            train_proc.terminate()
            return 1

        # Poll until training completes
        watcher.poll_until_done(training_process=train_proc, max_duration=args.timeout)

        # Wait for training to fully exit
        try:
            train_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.log("Training did not exit cleanly, terminating")
            train_proc.terminate()

    except KeyboardInterrupt:
        watcher.log("Interrupted, terminating training")
        train_proc.terminate()
        raise

    # Report
    report = watcher.generate_report()
    report["training_exit_code"] = train_proc.returncode

    print("\n" + "=" * 60)
    print("HEARTBEAT E2E TEST REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))

    # Validate
    failures = watcher.validate_contract()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("\nAll contract checks PASSED")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end test for heartbeat system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "logs",
        help="Logs directory (default: tdmpc2/logs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to watch (otherwise discovers newest)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum time to poll heartbeats (seconds)",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run full e2e test (launches training + watcher)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pendulum-swingup",
        help="Task for full test (default: pendulum-swingup)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Training steps for full test (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for full test",
    )

    args = parser.parse_args()

    if args.full_test:
        return run_full_test(args)
    else:
        return run_watcher_only(args)


if __name__ == "__main__":
    sys.exit(main())

