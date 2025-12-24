"""Tests for HeartbeatWriter."""

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest import TestCase
import importlib.util

# Import the heartbeat module directly to avoid triggering heavy dependencies
# (torch, numpy, etc.) that are imported via common/__init__.py.
# The heartbeat module itself has no external dependencies beyond stdlib.
spec = importlib.util.spec_from_file_location(
    "heartbeat", 
    Path(__file__).parent.parent / "common" / "heartbeat.py"
)
heartbeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heartbeat)
HeartbeatWriter = heartbeat.HeartbeatWriter
compute_work_hash = heartbeat.compute_work_hash


class TestComputeWorkHash(TestCase):
    """Tests for compute_work_hash function."""
    
    def test_deterministic(self):
        """Hash is deterministic for same inputs."""
        h1 = compute_work_hash("train", "walker-walk", 42)
        h2 = compute_work_hash("train", "walker-walk", 42)
        self.assertEqual(h1, h2)
    
    def test_different_task(self):
        """Different tasks produce different hashes."""
        h1 = compute_work_hash("train", "walker-walk", 42)
        h2 = compute_work_hash("train", "walker-run", 42)
        self.assertNotEqual(h1, h2)
    
    def test_different_seed(self):
        """Different seeds produce different hashes."""
        h1 = compute_work_hash("train", "walker-walk", 42)
        h2 = compute_work_hash("train", "walker-walk", 43)
        self.assertNotEqual(h1, h2)
    
    def test_different_kind(self):
        """Different kinds produce different hashes."""
        h1 = compute_work_hash("train", "walker-walk", 42)
        h2 = compute_work_hash("eval", "walker-walk", 42)
        self.assertNotEqual(h1, h2)
    
    def test_no_seed(self):
        """Hash works without seed."""
        h1 = compute_work_hash("train", "walker-walk")
        h2 = compute_work_hash("train", "walker-walk")
        self.assertEqual(h1, h2)
    
    def test_hash_length(self):
        """Hash is 16 characters (64-bit, reduced collision risk)."""
        h = compute_work_hash("train", "walker-walk", 42)
        self.assertEqual(len(h), 16)


class TestHeartbeatWriter(TestCase):
    """Tests for HeartbeatWriter class."""
    
    def setUp(self):
        """Create temp directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_write_atomic(self):
        """Test atomic write creates heartbeat.json."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run-123",
            kind="train",
            task="walker-walk",
            seed=42,
            enabled=True,
        )
        
        writer.write_now()
        
        hb_path = Path(self.temp_dir) / "heartbeat.json"
        self.assertTrue(hb_path.exists())
        
        with open(hb_path) as f:
            data = json.load(f)
        
        # Check required fields
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["run_id"], "test-run-123")
        self.assertEqual(data["kind"], "train")
        self.assertEqual(data["task"], "walker-walk")
        self.assertEqual(data["seed"], 42)
        self.assertIn("timestamp", data)
        self.assertIn("work_hash", data)
        self.assertEqual(data["progress"]["step"], 0)
        self.assertEqual(data["job"]["scheduler"], "lsf")
        self.assertIn("hostname", data["host"])
        self.assertIn("pid", data["host"])
    
    def test_update_step(self):
        """Test step updates are reflected in heartbeat."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=True,
        )
        
        writer.update_step(1000)
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        self.assertEqual(data["progress"]["step"], 1000)
    
    def test_step_non_decreasing(self):
        """Test step cannot decrease."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=True,
        )
        
        writer.update_step(1000)
        writer.update_step(500)  # Try to decrease
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        # Step should remain at 1000
        self.assertEqual(data["progress"]["step"], 1000)
    
    def test_update_checkpoint(self):
        """Test checkpoint info is included."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=True,
        )
        
        writer.update_checkpoint(5000, "/path/to/checkpoint.pt")
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        self.assertEqual(data["checkpoint"]["step"], 5000)
        self.assertEqual(data["checkpoint"]["path"], "/path/to/checkpoint.pt")
    
    def test_update_status(self):
        """Test status updates."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=True,
        )
        
        writer.update_status("stopping")
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        self.assertEqual(data["status"], "stopping")
    
    def test_disabled_no_write(self):
        """Test disabled writer doesn't create file."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=False,
        )
        
        writer.write_now()
        
        hb_path = Path(self.temp_dir) / "heartbeat.json"
        self.assertFalse(hb_path.exists())
    
    def test_context_manager(self):
        """Test context manager starts and stops correctly."""
        with HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            interval=0.1,  # Fast interval for testing
            enabled=True,
        ) as writer:
            # Heartbeat should be written on start
            hb_path = Path(self.temp_dir) / "heartbeat.json"
            self.assertTrue(hb_path.exists())
            
            # Update step and wait a bit
            writer.update_step(100)
            time.sleep(0.15)  # Wait for periodic update
        
        # After context, file should still exist with updated step
        with open(hb_path) as f:
            data = json.load(f)
        self.assertEqual(data["progress"]["step"], 100)
    
    def test_work_hash_included(self):
        """Test work_hash is included and matches compute_work_hash."""
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            seed=42,
            enabled=True,
        )
        
        expected_hash = compute_work_hash("train", "walker-walk", 42)
        self.assertEqual(writer.work_hash, expected_hash)
        
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        self.assertEqual(data["work_hash"], expected_hash)
    
    def test_lsf_job_id_from_env(self):
        """Test LSF job ID is read from environment."""
        os.environ["LSB_JOBID"] = "12345"
        try:
            writer = HeartbeatWriter(
                work_dir=self.temp_dir,
                run_id="test-run",
                kind="train",
                task="walker-walk",
                enabled=True,
            )
            
            writer.write_now()
            
            with open(Path(self.temp_dir) / "heartbeat.json") as f:
                data = json.load(f)
            
            self.assertEqual(data["job"]["job_id"], "12345")
        finally:
            del os.environ["LSB_JOBID"]
    
    def test_periodic_updates(self):
        """Test that periodic updates occur and update timestamp."""
        hb_path = Path(self.temp_dir) / "heartbeat.json"
        
        with HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            interval=0.05,  # 50ms interval for fast testing
            enabled=True,
        ) as writer:
            # Read initial heartbeat
            with open(hb_path) as f:
                data1 = json.load(f)
            ts1 = datetime.fromisoformat(data1["timestamp"])
            
            # Update step
            writer.update_step(500)
            
            # Wait for at least 2 intervals
            time.sleep(0.15)
            
            # Read again
            with open(hb_path) as f:
                data2 = json.load(f)
            ts2 = datetime.fromisoformat(data2["timestamp"])
            
            # Timestamp should have advanced
            self.assertGreater(ts2, ts1)
            # Step should be updated
            self.assertEqual(data2["progress"]["step"], 500)
    
    def test_concurrent_updates(self):
        """Test thread-safety of concurrent updates while writer is running."""
        hb_path = Path(self.temp_dir) / "heartbeat.json"
        errors = []
        
        def updater(writer, n):
            """Thread that rapidly updates state."""
            try:
                for i in range(100):
                    writer.update_step(n * 1000 + i)
                    writer.update_checkpoint(n * 1000 + i, f"/ckpt/{n}/{i}.pt")
                    writer.update_status("running" if i % 2 == 0 else "busy")
            except Exception as e:
                errors.append(e)
        
        with HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            interval=0.02,  # 20ms interval
            enabled=True,
        ) as writer:
            # Start multiple updater threads
            threads = [threading.Thread(target=updater, args=(writer, i)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Let a few more writes happen
            time.sleep(0.1)
        
        # No errors should have occurred
        self.assertEqual(errors, [])
        
        # File should be valid JSON with expected structure
        with open(hb_path) as f:
            data = json.load(f)
        
        # Basic structure checks
        self.assertIsInstance(data["progress"]["step"], int)
        self.assertGreaterEqual(data["progress"]["step"], 0)
        self.assertIn(data["status"], ["running", "busy", "stopping"])
        if "checkpoint" in data:
            self.assertIsInstance(data["checkpoint"]["step"], int)
    
    def test_job_id_empty_when_not_lsf(self):
        """Test job_id is empty string when LSB_JOBID not set."""
        # Ensure LSB_JOBID is not set
        os.environ.pop("LSB_JOBID", None)
        
        writer = HeartbeatWriter(
            work_dir=self.temp_dir,
            run_id="test-run",
            kind="train",
            task="walker-walk",
            enabled=True,
        )
        
        writer.write_now()
        
        with open(Path(self.temp_dir) / "heartbeat.json") as f:
            data = json.load(f)
        
        # job_id should be empty string (documented behavior)
        self.assertEqual(data["job"]["job_id"], "")
        self.assertEqual(data["job"]["scheduler"], "lsf")


if __name__ == "__main__":
    from unittest import main
    main()
