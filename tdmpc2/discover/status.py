#!/usr/bin/env python3
"""
CLI to show training status overview.

Usage:
  python status.py              # Show status
  python status.py --debug      # Show debug info about data
  python status.py --refresh    # Force refresh from wandb
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from discover.runs import discover, DEFAULT_LOGS_DIR, DEFAULT_WANDB_PROJECT
from discover.progress import attach_max_step, best_step_by_task


def main():
    parser = argparse.ArgumentParser(description="Show training status")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--refresh", action="store_true", help="Force refresh (ignore cache)")
    parser.add_argument("--limit", type=int, default=None, help="Limit wandb runs")
    parser.add_argument("--target", type=int, default=5_000_000, help="Target steps")
    args = parser.parse_args()
    
    print(f"Logs dir: {DEFAULT_LOGS_DIR}")
    print(f"Wandb project: {DEFAULT_WANDB_PROJECT}")
    print()
    
    # Load data
    df = discover(DEFAULT_LOGS_DIR, DEFAULT_WANDB_PROJECT, args.limit)
    
    print(f"\n=== RAW DATA ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if args.debug and not df.empty:
        print(f"\nSample rows:")
        print(df[['task', 'status', 'found_in']].head(10).to_string())
        
        # Check summary column
        if 'summary' in df.columns:
            print(f"\n=== SUMMARY COLUMN DEBUG ===")
            non_null = df[df['summary'].notna()]
            print(f"Rows with summary: {len(non_null)}")
            if len(non_null) > 0:
                sample = non_null.iloc[0]['summary']
                print(f"Sample summary type: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"Sample summary keys: {list(sample.keys())[:10]}")
                    if '_step' in sample:
                        print(f"Sample _step: {sample['_step']}")
                    else:
                        print("WARNING: '_step' not in summary!")
                else:
                    print(f"Sample summary value: {sample[:100] if isinstance(sample, str) else sample}")
        
        # Check ckpt_step column
        if 'ckpt_step' in df.columns:
            print(f"\n=== CKPT_STEP COLUMN DEBUG ===")
            non_zero = df[df['ckpt_step'] > 0] if 'ckpt_step' in df.columns else None
            if non_zero is not None:
                print(f"Rows with ckpt_step > 0: {len(non_zero)}")
    
    # Attach max_step and get best per task
    print(f"\n=== ANALYSIS ===")
    df_with_step = attach_max_step(df)
    
    if args.debug:
        print(f"max_step column stats:")
        print(f"  non-null: {df_with_step['max_step'].notna().sum()}")
        print(f"  > 0: {(df_with_step['max_step'] > 0).sum()}")
        print(f"  max: {df_with_step['max_step'].max()}")
    
    best_df = best_step_by_task(df)
    print(f"Unique tasks: {len(best_df)}")
    
    # Calculate progress
    target = args.target
    if 'max_step' not in best_df.columns:
        best_df = attach_max_step(best_df)
    
    best_df['progress'] = best_df['max_step'].fillna(0) / target
    
    completed = (best_df['max_step'] >= target).sum()
    in_progress = ((best_df['max_step'] > 0) & (best_df['max_step'] < target)).sum()
    not_started = (best_df['max_step'].fillna(0) == 0).sum()
    
    print(f"\n=== TRAINING STATUS ===")
    print(f"Target: {target:,} steps")
    print(f"  ✅ Completed:    {completed} ({100*completed/len(best_df):.1f}%)")
    print(f"  🔄 In Progress:  {in_progress} ({100*in_progress/len(best_df):.1f}%)")
    print(f"  ❌ Not Started:  {not_started} ({100*not_started/len(best_df):.1f}%)")
    
    if args.debug and completed > 0:
        print(f"\nCompleted tasks:")
        completed_tasks = best_df[best_df['max_step'] >= target]['task'].tolist()
        for t in completed_tasks[:10]:
            print(f"  - {t}")
        if len(completed_tasks) > 10:
            print(f"  ... and {len(completed_tasks) - 10} more")
    
    if args.debug and in_progress > 0:
        print(f"\nIn progress tasks (sample):")
        ip_df = best_df[(best_df['max_step'] > 0) & (best_df['max_step'] < target)]
        for _, row in ip_df.head(5).iterrows():
            print(f"  - {row['task']}: {row['max_step']:,.0f} ({100*row['progress']:.1f}%)")


if __name__ == "__main__":
    main()

