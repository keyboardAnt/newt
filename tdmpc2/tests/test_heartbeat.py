"""Tests for HeartbeatWriter."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import TestCase
import importlib.util

# Import the heartbeat module directly to avoid torch dependencies in common/__init__.py
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
        """Hash is 12 characters (short hash)."""
        h = compute_work_hash("train", "walker-walk", 42)
        self.assertEqual(len(h), 12)


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


if __name__ == "__main__":
    import unittest
    unittest.main()
