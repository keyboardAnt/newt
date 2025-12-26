from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "tdmpc2" / "logs"

    print("Scanning for run_info.yaml files...")
    run_infos = list(logs_dir.glob("**/run_info.yaml"))
    print(f"Found {len(run_infos)} run_info.yaml files.")

    print("\nChecking wandb location relative to run directory (where run_info.yaml is)...")
    relative_depths: dict[int, int] = {}

    for ri in run_infos:
        run_dir = ri.parent

        # Find wandb dirs inside this run_dir. We care about their depth relative to run_dir:
        # - depth 0: run_dir/wandb
        # - depth 1: run_dir/<something>/wandb
        wandb_dirs = list(run_dir.glob("**/wandb"))
        for w in wandb_dirs:
            if not w.is_dir():
                continue

            rel_path = w.relative_to(run_dir)
            depth = len(rel_path.parts) - 1  # ('wandb',) -> 0
            relative_depths[depth] = relative_depths.get(depth, 0) + 1

    print("\n'wandb' directory depth relative to run_info.yaml location:")
    for depth, count in sorted(relative_depths.items()):
        print(f"Depth {depth}: {count} runs (e.g. depth=0 means run_dir/wandb)")

    print("\nVerifying if any are missed by the new patterns:")
    print("Pattern 1: run_dir/wandb/... (Depth 0)")
    print("Pattern 2: run_dir/*/wandb/... (Depth 1)")

    missed_count = 0
    for depth, count in relative_depths.items():
        if depth > 1:
            missed_count += count
            print(f"WARNING: Depth {depth} will be MISSED! ({count} runs)")

    if missed_count == 0:
        print("\nSUCCESS: All wandb directories are covered by depth 0 and 1 patterns.")
        return 0

    print(f"\nFAILURE: {missed_count} wandb directories would be missed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


