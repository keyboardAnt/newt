from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "tdmpc2" / "logs"

    print("Scanning for wandb video directories...")
    deep_videos = list(logs_dir.glob("**/wandb/run-*/files/media/videos/**/*.mp4"))

    print(f"Found {len(deep_videos)} video files.")

    # Analyze depths (relative to logs_dir)
    depths: dict[int, int] = {}
    for v in deep_videos:
        rel_path = v.relative_to(logs_dir)
        parts = rel_path.parts
        try:
            wandb_index = parts.index("wandb")
            depths[wandb_index] = depths.get(wandb_index, 0) + 1
        except ValueError:
            # Shouldn't happen given the glob, but keep robust
            pass

    print("\nVideo locations by 'wandb' directory depth (0 = logs/wandb, 1 = logs/task/wandb, etc.):")
    for depth, count in sorted(depths.items()):
        print(f"Depth {depth}: {count} videos")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


