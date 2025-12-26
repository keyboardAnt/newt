"""Backward-compatible shim.

Implementation moved to `discover.cleanup.wandb_models`.
"""

from .cleanup.wandb_models import (  # noqa: F401
    CleanupPlan,
    ParsedModelArtifactName,
    apply_cleanup_plan,
    iter_model_collections,
    plan_cleanup_latest_checkpoint_per_expert,
    print_cleanup_plan,
)

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


_STEP_TOKEN_RE = re.compile(r"^[0-9][0-9_]*$")
_SEED_TOKEN_RE = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class ParsedModelArtifactName:
    """Parsed checkpoint artifact name.

    Training logs checkpoints as separate W&B *artifact names* (collections), not versions:
      <task>-<exp_name>-<seed>-<step_with_underscores>
    """

    expert_key: str  # everything up to and including seed: "<task>-<exp_name>-<seed>"
    step: int  # parsed from final token


def _parse_model_artifact_name(name: str) -> Optional[ParsedModelArtifactName]:
    # Sometimes name is "collection:vX" (Artifact.name) and sometimes just "collection".
    name = str(name).split(":", 1)[0]
    # Expected from tdmpc2/common/logger.py:
    #   artifact_name = cfg_to_group(cfg) + "-" + seed + "-" + identifier
    # where identifier is f"{step:,}".replace(",", "_")
    parts = str(name).rsplit("-", 2)
    if len(parts) != 3:
        return None
    prefix, seed_token, step_token = parts
    if not _SEED_TOKEN_RE.match(seed_token):
        return None
    if not _STEP_TOKEN_RE.match(step_token):
        return None
    step = int(step_token.replace("_", ""))
    expert_key = f"{prefix}-{seed_token}"
    return ParsedModelArtifactName(expert_key=expert_key, step=step)


def _split_entity_project(project_path: str) -> Tuple[str, str]:
    if not project_path or "/" not in project_path:
        raise ValueError(f"Expected wandb project path 'entity/project', got: {project_path!r}")
    entity, project = project_path.split("/", 1)
    if not entity or not project:
        raise ValueError(f"Expected wandb project path 'entity/project', got: {project_path!r}")
    return entity, project


def iter_model_collections(
    project_path: str,
    *,
    collection_name_regex: Optional[str] = None,
    exact_collections: Optional[Iterable[str]] = None,
    max_collections: Optional[int] = None,
    progress_every: int = 200,
) -> Iterator[object]:
    """Yield W&B ArtifactCollection objects of type 'model' for a given entity/project.

    Notes:
    - In wandb==0.22.x, iterating `Api.artifacts(...)` can fail to parse responses depending on backend.
      `Api.artifact_type(...).collections()` is more reliable and also exposes per-collection versions.
    """
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        sys.stderr.write("wandb is required. Install with `pip install wandb`.\n")
        raise SystemExit(1) from exc

    name_re = re.compile(collection_name_regex) if collection_name_regex else None
    api = wandb.Api(timeout=60)
    at = api.artifact_type("model", project_path)

    if exact_collections:
        yielded = 0
        for col_name in exact_collections:
            col_name = str(col_name).strip()
            if not col_name:
                continue
            if name_re and not name_re.search(col_name):
                continue
            # Fast path: direct fetch, no project-wide scanning.
            try:
                yield api.artifact_collection("model", f"{project_path}/{col_name}")
                yielded += 1
            except Exception as e:
                sys.stderr.write(f"[cleanup-models] Failed to fetch collection {col_name!r}: {e}\n")
                continue
            if max_collections is not None and yielded >= max_collections:
                break
        return
    scanned = 0
    yielded = 0
    for col in at.collections():
        scanned += 1
        if progress_every and scanned % int(progress_every) == 0:
            sys.stderr.write(f"\rScanning model artifact collections... {scanned}")
            sys.stderr.flush()

        try:
            col_name = str(getattr(col, "name", "") or "")
        except Exception:
            continue

        if name_re and not name_re.search(col_name):
            continue

        yield col
        yielded += 1
        if max_collections is not None and yielded >= max_collections:
            break

    if scanned:
        sys.stderr.write(f"\rScanning model artifact collections... {scanned} ✓\n")
        sys.stderr.flush()


def iter_model_artifacts(
    project_path: str,
    *,
    collection_name_regex: Optional[str] = None,
    exact_collections: Optional[Iterable[str]] = None,
    max_collections: Optional[int] = None,
) -> Iterator[object]:
    """Yield W&B Artifact objects (versions) for model artifacts in a project."""
    for col in iter_model_collections(
        project_path,
        collection_name_regex=collection_name_regex,
        exact_collections=exact_collections,
        max_collections=max_collections,
    ):
        for art in col.artifacts():
            yield art


def _parse_version_int(version: str) -> Optional[int]:
    # Expect "v0", "v12", etc.
    s = str(version).strip()
    if not s.startswith("v"):
        return None
    tail = s[1:]
    if not tail.isdigit():
        return None
    return int(tail)


@dataclass
class CleanupPlan:
    keep: List[object]
    delete: List[object]
    skipped: int
    by_expert_keep_step: Dict[str, int]


def plan_cleanup_latest_checkpoint_per_expert(
    *,
    project_path: str,
    protect_aliases: Iterable[str] = ("latest", "best", "prod", "production", "staging"),
    name_regex: Optional[str] = None,
    exact_collections: Optional[Iterable[str]] = None,
    max_collections: Optional[int] = None,
) -> CleanupPlan:
    """Create a cleanup plan that keeps only max-step checkpoint per expert_key.

    - expert_key is "<task>-<exp_name>-<seed>" inferred from artifact name.
    - step is inferred from the last name token (digits + underscores).
    """
    protect_aliases_set = {a.strip() for a in protect_aliases if str(a).strip()}
    # IMPORTANT: apply regex at *collection name* level to avoid downloading versions for everything.
    name_re = re.compile(name_regex) if name_regex else None

    # best[expert_key] = (best_step, best_artifact)
    best: Dict[str, int] = {}
    # Group artifact *versions* by collection name (collection contains step)
    by_collection: Dict[str, List[object]] = {}
    skipped = 0

    for col in iter_model_collections(
        project_path,
        collection_name_regex=name_regex,
        exact_collections=exact_collections,
        max_collections=max_collections,
    ):
        try:
            collection_name = str(getattr(col, "name", "") or "")
        except Exception:
            continue

        if name_re and not name_re.search(collection_name):
            # Redundant (iter_model_collections already filters), but keep for safety.
            continue

        parsed = _parse_model_artifact_name(collection_name)
        if parsed is None:
            continue

        cur_step = best.get(parsed.expert_key)
        if cur_step is None or parsed.step > cur_step:
            best[parsed.expert_key] = parsed.step

        # Only fetch versions for collections that passed the filter.
        try:
            versions = list(col.artifacts())
        except Exception:
            skipped += 1
            continue
        by_collection[collection_name] = versions

    keep: List[object] = []
    delete: List[object] = []

    by_expert_keep_step: Dict[str, int] = dict(best)

    # Decide keep/delete at version granularity.
    for collection_name, versions in by_collection.items():
        parsed = _parse_model_artifact_name(collection_name)
        if parsed is None:
            skipped += len(versions)
            continue
        max_step = by_expert_keep_step.get(parsed.expert_key)
        if max_step is None:
            skipped += len(versions)
            continue

        # Helper: keep any version with protected aliases.
        protected: Set[object] = set()
        if protect_aliases_set:
            for art in versions:
                try:
                    aliases = list(getattr(art, "aliases", []) or [])
                except Exception:
                    aliases = []
                if any(a in protect_aliases_set for a in aliases):
                    protected.add(art)

        if parsed.step != max_step:
            # Old step: delete all versions unless protected.
            for art in versions:
                (keep if art in protected else delete).append(art)
            continue

        # Max step: keep only first and last version (by numeric vN), plus any protected aliases.
        indexed: List[Tuple[int, object]] = []
        for art in versions:
            v = _parse_version_int(getattr(art, "version", ""))
            if v is None:
                # Can't reason about order; keep it unless it's clearly safe to delete.
                protected.add(art)
            else:
                indexed.append((v, art))

        if not indexed:
            # No parseable versions; keep everything (better safe than sorry).
            keep.extend(list(set(versions)))
            continue

        indexed.sort(key=lambda t: t[0])
        first_art = indexed[0][1]
        last_art = indexed[-1][1]
        for _, art in indexed:
            if art in protected or art is first_art or art is last_art:
                keep.append(art)
            else:
                delete.append(art)

    return CleanupPlan(keep=keep, delete=delete, skipped=skipped, by_expert_keep_step=by_expert_keep_step)


def _format_int(n: int) -> str:
    return f"{int(n):,}"


def print_cleanup_plan(plan: CleanupPlan, *, out=sys.stdout) -> None:
    out.write("\n")
    out.write("=" * 88 + "\n")
    out.write(f"{'WANDB MODEL ARTIFACT CLEANUP (PLAN)':^88}\n")
    out.write("=" * 88 + "\n")
    out.write(f"Experts found:         {_format_int(len(plan.by_expert_keep_step))}\n")
    out.write(f"Artifacts to KEEP:     {_format_int(len(plan.keep))}\n")
    out.write(f"Artifacts to DELETE:   {_format_int(len(plan.delete))}\n")
    out.write(f"Artifacts skipped:     {_format_int(plan.skipped)}  (name not parseable / filtered)\n")
    out.write("-" * 88 + "\n")

    # Show a small sample of deletions (names only) for sanity.
    keep_sample = plan.keep[:25]
    if keep_sample:
        out.write("Keep candidates (sample):\n")
        for art in keep_sample:
            name = getattr(art, "name", "?")
            version = getattr(art, "version", "")
            if version and f":{version}" not in str(name):
                out.write(f"  + {name}:{version}\n")
            else:
                out.write(f"  + {name}\n")
        if len(plan.keep) > len(keep_sample):
            out.write(f"  ... and {_format_int(len(plan.keep) - len(keep_sample))} more\n")
        out.write("-" * 88 + "\n")

    sample = plan.delete[:25]
    if sample:
        out.write("Delete candidates (sample):\n")
        for art in sample:
            name = getattr(art, "name", "?")
            version = getattr(art, "version", "")
            # Artifact.name often already includes ":vN"; avoid double-appending.
            if version and f":{version}" not in str(name):
                out.write(f"  - {name}:{version}\n")
            else:
                out.write(f"  - {name}\n")
        if len(plan.delete) > len(sample):
            out.write(f"  ... and {_format_int(len(plan.delete) - len(sample))} more\n")
    out.write("=" * 88 + "\n")


def apply_cleanup_plan(
    plan: CleanupPlan,
    *,
    max_delete: Optional[int] = 500,
    force: bool = False,
    out=sys.stdout,
) -> int:
    """Delete artifacts in the plan (returns number deleted)."""
    to_delete = list(plan.delete)
    if max_delete is not None and len(to_delete) > max_delete:
        raise RuntimeError(
            f"Refusing to delete {len(to_delete)} artifacts (> max_delete={max_delete}). "
            f"Re-run with --max-delete {len(to_delete)} (or larger) if you're sure."
        )

    deleted = 0
    for art in to_delete:
        name = getattr(art, "name", "?")
        version = getattr(art, "version", "")
        try:
            # delete_aliases=True is important if any alias exists (else delete may fail).
            # Some W&B installations require a "delete" boolean or other signature; keep this minimal.
            art.delete(delete_aliases=True)
            deleted += 1
            if version and f":{version}" not in str(name):
                out.write(f"Deleted: {name}:{version}\n")
            else:
                out.write(f"Deleted: {name}\n")
        except Exception as e:
            if not force:
                raise
            if version and f":{version}" not in str(name):
                out.write(f"FAILED (continuing due to --force): {name}:{version} :: {e}\n")
            else:
                out.write(f"FAILED (continuing due to --force): {name} :: {e}\n")
    return deleted


