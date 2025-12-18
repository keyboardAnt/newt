"""
Automatic UTD (Update-to-Data ratio) scaling based on GPU utilization.

The key insight: if GPU utilization is low but memory is available,
we can safely increase UTD to do more compute per environment step.

## Reproducibility & Debugging

This module maintains a complete history of all UTD changes:
- `utd_history`: List of (step, utd_value) tuples for exact reproducibility
- `adjustment_log`: Detailed log of every adjustment with reasons and metrics
- Both are saved to `auto_utd_log.json` in the run directory

## Validation

To validate without risking production runs:
1. Enable with `auto_utd=true auto_utd_dry_run=true` - logs what would happen without changing UTD
2. Set conservative limits: `auto_utd_max=0.2` to cap maximum UTD
3. Monitor W&B metrics: `auto_utd/*` panel shows all relevant info
"""
import json
import torch
from datetime import datetime
from pathlib import Path
from time import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List
from termcolor import colored

# Try to import pynvml for GPU utilization metrics
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
except Exception as e:
    # pynvml installed but init failed
    print(colored(f'[Auto-UTD] pynvml init failed: {e}. GPU utilization monitoring disabled.', 'yellow'))
    PYNVML_AVAILABLE = False


@dataclass
class UTDAdjustment:
    """Record of a single UTD adjustment for debugging and reproducibility."""
    step: int
    timestamp: str
    old_utd: float
    new_utd: float
    reason: str
    update_fraction: float
    memory_fraction: float
    memory_allocated_gb: float
    memory_total_gb: float
    
    def to_dict(self):
        return asdict(self)


@dataclass  
class UTDSnapshot:
    """Snapshot of UTD value at a given step for reproducibility."""
    step: int
    utd: float


class AdaptiveUTD:
    """
    Adaptively scale UTD based on the ratio of update time to total step time.
    
    If updates are fast relative to env stepping, we have GPU headroom.
    
    Features:
    - Complete history logging for reproducibility
    - Dry-run mode for validation
    - Memory safety checks with early warning
    - Detailed adjustment logs for debugging
    """
    
    def __init__(
        self,
        initial_utd: float = 0.075,
        min_utd: float = 0.05,
        max_utd: float = 0.5,
        target_update_fraction: float = 0.6,  # Target: 60% of time in updates
        adjustment_interval: int = 1000,
        smoothing_window: int = 100,
        memory_headroom: float = 0.15,  # Keep 15% memory free
        memory_warning_threshold: float = 0.88,  # Warn at 88%
        mode: str = "off",  # "off", "dry_run", or "on"
        work_dir: Optional[Path] = None,
        rank: int = 0,
    ):
        self.initial_utd = initial_utd
        self.utd = initial_utd
        self.min_utd = min_utd
        self.max_utd = max_utd
        self.target_fraction = target_update_fraction
        self.adjustment_interval = adjustment_interval
        self.memory_headroom = memory_headroom
        self.memory_warning_threshold = memory_warning_threshold
        
        # Parse mode
        self.mode = mode.lower()
        self.enabled = self.mode in ("on", "dry_run")
        self.dry_run = self.mode == "dry_run"
        self.work_dir = Path(work_dir) if work_dir else None
        self.rank = rank
        
        # Timing tracking
        self._update_times = deque(maxlen=smoothing_window)
        self._step_times = deque(maxlen=smoothing_window)
        self._last_step_time = None
        self._steps_since_adjust = 0
        self._current_step = 0
        
        # For logging
        self.last_update_fraction = 0.0
        self.last_memory_fraction = 0.0
        
        # === REPRODUCIBILITY: Complete history ===
        self.utd_history: List[UTDSnapshot] = [UTDSnapshot(step=0, utd=initial_utd)]
        self.adjustment_log: List[UTDAdjustment] = []
        
        # === VALIDATION: Memory tracking for early OOM detection ===
        self._memory_samples = deque(maxlen=50)
        self._memory_warnings = 0
        self._oom_near_misses = 0
        
        # === OBSERVABILITY: Heartbeat and GPU monitoring ===
        self._check_count = 0
        self._heartbeat_interval = 10  # Print status every N checks
        self._gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
            except Exception as e:
                if rank == 0:
                    print(colored(f'[Auto-UTD] Failed to get GPU handle: {e}', 'yellow'))
        elif rank == 0 and self.enabled:
            print(colored('[Auto-UTD] pynvml not available. Install with: pip install pynvml', 'yellow'))
    
    def set_step(self, step: int):
        """Update current step counter."""
        self._current_step = step
    
    def start_step(self):
        """Call at the start of each training step."""
        now = time()
        if self._last_step_time is not None:
            self._step_times.append(now - self._last_step_time)
        self._last_step_time = now
    
    def record_update_time(self, duration: float):
        """Record how long an update took."""
        self._update_times.append(duration)
        self._steps_since_adjust += 1
    
    def get_memory_info(self) -> tuple:
        """Get current GPU memory usage (fraction, allocated_gb, total_gb)."""
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total, allocated / 1e9, total / 1e9
        except:
            return 0.5, 0.0, 0.0  # Fallback
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU compute utilization (0-1)."""
        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                return util.gpu / 100.0
            except Exception:
                pass
        return 0.0  # Fallback if not available
    
    def check_memory_safety(self) -> dict:
        """
        Check memory safety and return warnings/status.
        Useful for early OOM detection.
        """
        mem_frac, mem_alloc, mem_total = self.get_memory_info()
        self._memory_samples.append(mem_frac)
        
        status = {'safe': True, 'warning': None}
        
        # Check for memory growth trend
        if len(self._memory_samples) >= 10:
            recent = list(self._memory_samples)[-10:]
            older = list(self._memory_samples)[:10] if len(self._memory_samples) >= 20 else recent
            growth = sum(recent) / len(recent) - sum(older) / len(older)
            
            if growth > 0.05:  # 5% growth
                status['warning'] = f'Memory growing: +{growth:.1%} trend'
        
        # Check thresholds
        if mem_frac > 0.95:
            status['safe'] = False
            status['warning'] = f'CRITICAL: Memory at {mem_frac:.1%}'
            self._oom_near_misses += 1
        elif mem_frac > self.memory_warning_threshold:
            self._memory_warnings += 1
            status['warning'] = f'High memory: {mem_frac:.1%}'
        
        return status
    
    def maybe_adjust(self) -> dict:
        """
        Check if UTD should be adjusted. Returns adjustment info for logging.
        """
        if not self.enabled:
            return {}
        
        if self._steps_since_adjust < self.adjustment_interval:
            return {}
        
        if len(self._update_times) < 10 or len(self._step_times) < 10:
            return {}
        
        self._steps_since_adjust = 0
        self._check_count += 1
        
        # Calculate time fractions
        avg_update_time = sum(self._update_times) / len(self._update_times)
        avg_step_time = sum(self._step_times) / len(self._step_times)
        
        if avg_step_time <= 0:
            return {}
        
        update_fraction = avg_update_time / avg_step_time
        mem_frac, mem_alloc, mem_total = self.get_memory_info()
        gpu_util = self.get_gpu_utilization()
        
        self.last_update_fraction = update_fraction
        self.last_memory_fraction = mem_frac
        self._last_gpu_util = gpu_util
        self._last_step_time_avg = avg_step_time
        
        old_utd = self.utd
        reason = None
        
        # Only increase if we have memory headroom
        has_memory_headroom = mem_frac < (1.0 - self.memory_headroom)
        
        if update_fraction < self.target_fraction * 0.7 and has_memory_headroom:
            # GPU underutilized and memory available - increase UTD
            scale = min(1.3, self.target_fraction / max(update_fraction, 0.1))
            new_utd = min(self.utd * scale, self.max_utd)
            reason = 'underutilized'
        elif update_fraction > self.target_fraction * 1.3:
            # Updates taking too long - decrease UTD
            scale = max(0.8, self.target_fraction / update_fraction)
            new_utd = max(self.utd * scale, self.min_utd)
            reason = 'overloaded'
        elif mem_frac > 0.9:
            # Memory pressure - decrease UTD
            new_utd = max(self.utd * 0.8, self.min_utd)
            reason = 'memory_pressure'
        else:
            new_utd = self.utd
        
        # === OBSERVABILITY: Heartbeat logging ===
        if self.rank == 0 and self._check_count % self._heartbeat_interval == 0:
            status = 'stable' if new_utd == old_utd else reason
            dry_tag = '[DRY-RUN] ' if self.dry_run else ''
            print(colored(
                f'[Auto-UTD] {dry_tag}step {self._current_step:,}: '
                f'UTD={self.utd:.4f}, update_frac={update_fraction:.0%}, '
                f'mem={mem_frac:.0%}, gpu={gpu_util:.0%} ({status})',
                'cyan'
            ))
        
        # Only log if there's a change
        if new_utd != old_utd:
            adjustment = UTDAdjustment(
                step=self._current_step,
                timestamp=datetime.now().isoformat(),
                old_utd=old_utd,
                new_utd=new_utd,
                reason=reason,
                update_fraction=update_fraction,
                memory_fraction=mem_frac,
                memory_allocated_gb=mem_alloc,
                memory_total_gb=mem_total,
            )
            self.adjustment_log.append(adjustment)
            
            # Print adjustment (always, not just on heartbeat)
            if self.rank == 0:
                dry_tag = '[DRY-RUN] ' if self.dry_run else ''
                color = 'yellow' if reason in ('overloaded', 'memory_pressure') else 'green'
                print(colored(
                    f'[Auto-UTD] {dry_tag}{reason}: UTD {old_utd:.4f} → {new_utd:.4f} '
                    f'(update_frac={update_fraction:.0%}, mem={mem_frac:.0%})',
                    color
                ))
            
            # Apply change (unless dry run)
            if not self.dry_run:
                self.utd = new_utd
                self.utd_history.append(UTDSnapshot(step=self._current_step, utd=new_utd))
            
            # Auto-save log periodically
            if len(self.adjustment_log) % 5 == 0:
                self.save_log()
            
            return adjustment.to_dict()
        
        return {}
    
    def get_metrics(self) -> dict:
        """Return current metrics for W&B logging."""
        mem_frac, mem_alloc, mem_total = self.get_memory_info()
        step_time = getattr(self, '_last_step_time_avg', 0.0)
        
        metrics = {
            # Core metrics
            'auto_utd/utd': self.utd,
            'auto_utd/utd_ratio_vs_initial': self.utd / self.initial_utd,
            'auto_utd/update_fraction': self.last_update_fraction,
            
            # GPU metrics (only if pynvml available)
            'auto_utd/step_time_seconds': step_time,
            
            # Memory metrics  
            'auto_utd/memory_fraction': mem_frac,
            'auto_utd/memory_allocated_gb': mem_alloc,
            
            # Safety metrics
            'auto_utd/memory_warnings': self._memory_warnings,
            'auto_utd/oom_near_misses': self._oom_near_misses,
            
            # History metrics
            'auto_utd/num_adjustments': len(self.adjustment_log),
            'auto_utd/dry_run': int(self.dry_run),
        }
        
        # Only log GPU utilization if pynvml is available (always get fresh value)
        if self._gpu_handle is not None:
            metrics['auto_utd/gpu_utilization'] = self.get_gpu_utilization()
        
        return metrics
    
    def print_config(self):
        """Print active thresholds at startup for transparency."""
        if self.rank != 0:
            return
        
        mode_display = {'on': 'ENABLED', 'dry_run': 'DRY-RUN', 'off': 'OFF'}.get(self.mode, self.mode.upper())
        print(colored(f'\n[Auto-UTD] Configuration ({mode_display}):', 'cyan', attrs=['bold']))
        print(f'  UTD range:        {self.min_utd:.3f} - {self.max_utd:.3f} (initial: {self.initial_utd:.3f})')
        print(f'  Target update%:   {self.target_fraction:.0%}')
        print(f'  Increase when:    update_frac < {self.target_fraction * 0.7:.0%} (underutilized)')
        print(f'  Decrease when:    update_frac > {self.target_fraction * 1.3:.0%} (overloaded)')
        print(f'  Memory headroom:  {self.memory_headroom:.0%} (decrease if mem > {1 - self.memory_headroom:.0%})')
        print(f'  Check interval:   every {self.adjustment_interval:,} steps')
        print(f'  Heartbeat:        every {self._heartbeat_interval * self.adjustment_interval:,} steps')
        gpu_status = '✓ enabled' if self._gpu_handle is not None else '✗ disabled (install pynvml)'
        print(f'  GPU util monitor: {gpu_status}\n')
    
    def get_effective_config(self) -> dict:
        """
        Get the effective UTD configuration for reproducibility.
        This should be saved with checkpoints.
        """
        return {
            'auto_utd_mode': self.mode,
            'auto_utd_initial': self.initial_utd,
            'auto_utd_current': self.utd,
            'auto_utd_min': self.min_utd,
            'auto_utd_max': self.max_utd,
            'auto_utd_target_fraction': self.target_fraction,
            'auto_utd_num_adjustments': len(self.adjustment_log),
        }
    
    def save_log(self):
        """Save complete adjustment log to disk for reproducibility."""
        if self.work_dir is None or self.rank != 0:
            return
        
        log_path = self.work_dir / 'auto_utd_log.json'
        log_data = {
            'config': self.get_effective_config(),
            'utd_history': [asdict(s) for s in self.utd_history],
            'adjustments': [a.to_dict() for a in self.adjustment_log],
            'final_utd': self.utd,
            'saved_at': datetime.now().isoformat(),
        }
        
        try:
            log_path.write_text(json.dumps(log_data, indent=2))
        except Exception as e:
            print(f'[Auto-UTD] Failed to save log: {e}')
    
    def get_summary(self) -> str:
        """Get a human-readable summary of UTD changes."""
        if not self.adjustment_log:
            return f"No UTD adjustments made. UTD remained at {self.utd:.4f}"
        
        lines = [
            f"Auto-UTD Summary:",
            f"  Initial UTD: {self.initial_utd:.4f}",
            f"  Final UTD:   {self.utd:.4f} ({self.utd/self.initial_utd:.1f}x)",
            f"  Adjustments: {len(self.adjustment_log)}",
            f"  Memory warnings: {self._memory_warnings}",
            f"  OOM near-misses: {self._oom_near_misses}",
            "",
            "Adjustment History:",
        ]
        
        for adj in self.adjustment_log[-10:]:  # Last 10
            lines.append(
                f"  Step {adj.step:>10}: {adj.old_utd:.4f} → {adj.new_utd:.4f} "
                f"({adj.reason}, mem={adj.memory_fraction:.1%})"
            )
        
        if len(self.adjustment_log) > 10:
            lines.append(f"  ... and {len(self.adjustment_log) - 10} more")
        
        return "\n".join(lines)

