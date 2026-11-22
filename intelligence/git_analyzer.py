"""
Git File Analyzer

Extracts per-file commit history from a local git repo.
Used by the risk scorer to answer: "How often has this file changed recently?"

Data collected:
  - commit count per file in the last N days (default 90)
  - last modified date per file

This is the "Recent changes" signal in the risk score breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Optional

try:
    from git import Repo, InvalidGitRepositoryError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

from .analyzer import get_parser, get_symbol_locations
from .models import EXTENSION_TO_LANGUAGE, SymbolLocation


@dataclass(frozen=True)
class DeploymentDiffLine:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class DeploymentDiffFile:
    path: str
    status: str
    changed_lines: tuple[DeploymentDiffLine, ...]


@dataclass(frozen=True)
class DeploymentSymbolMatch:
    path: str
    symbol: SymbolLocation
    changed_lines: tuple[DeploymentDiffLine, ...]


@dataclass(frozen=True)
class DeploymentDiffReport:
    target_commit: str
    stable_commit: str
    files: tuple[DeploymentDiffFile, ...]
    symbol_matches: tuple[DeploymentSymbolMatch, ...] = ()


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_DIFF_FILE_RE = re.compile(r'^\+\+\+ b/(.+)$')


def _parse_target_diff(diff_text: str) -> tuple[DeploymentDiffFile, ...]:
    files: list[DeploymentDiffFile] = []
    path: Optional[str] = None
    status = "modified"
    changed_lines: list[DeploymentDiffLine] = []

    def finish_file() -> None:
        nonlocal path, status, changed_lines
        if path:
            files.append(DeploymentDiffFile(path, status, tuple(changed_lines)))
        path = None
        status = "modified"
        changed_lines = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            finish_file()
        elif line.startswith("new file mode "):
            status = "added"
        elif line.startswith("deleted file mode "):
            status = "deleted"
        elif line.startswith("rename to "):
            status = "renamed"
            path = line[len("rename to "):].strip()
        elif line.startswith("+++ b/"):
            match = _DIFF_FILE_RE.match(line)
            path = match.group(1) if match else None
        elif line.startswith("@@"):
            match = _HUNK_RE.match(line)
            if match and path:
                start = int(match.group(1))
                count = int(match.group(2) or "1")
                if count:
                    changed_lines.append(DeploymentDiffLine(start, start + count - 1))

    finish_file()
    return tuple(files)


def _lines_overlap(symbol: SymbolLocation, line_range: DeploymentDiffLine) -> bool:
    return symbol.start_line <= line_range.end_line and symbol.end_line >= line_range.start_line


def cross_reference_symbol_targets(
    diff_files: Iterable[DeploymentDiffFile],
    symbol_targets: Mapping[str, Iterable[SymbolLocation]],
) -> tuple[DeploymentSymbolMatch, ...]:
    """Return target symbols whose source ranges overlap changed target lines."""
    matches: list[DeploymentSymbolMatch] = []
    targets_by_path = {
        path: tuple(locations)
        for path, locations in symbol_targets.items()
    }

    for changed_file in diff_files:
        for symbol in targets_by_path.get(changed_file.path, ()):
            overlaps = tuple(
                line_range
                for line_range in changed_file.changed_lines
                if _lines_overlap(symbol, line_range)
            )
            if overlaps:
                matches.append(DeploymentSymbolMatch(changed_file.path, symbol, overlaps))

    return tuple(matches)


def analyze_deployment_diff(
    repo_path: str,
    target_commit_sha: str,
    stable_commit_sha: str,
) -> DeploymentDiffReport:
    """Compare a deployed commit with a stable baseline and find touched symbols.

    The diff is anchored to the target commit.  Existing history helpers remain
    independent of this deployment-specific analysis.
    """
    if not GIT_AVAILABLE:
        raise RuntimeError("GitPython is required for deployment diff analysis")

    repo = Repo(repo_path)
    try:
        target = repo.commit(target_commit_sha)
        stable = repo.commit(stable_commit_sha)
    except Exception as exc:
        raise ValueError("target and stable commit SHAs must resolve to repository commits") from exc
    target_sha = target.hexsha
    stable_sha = stable.hexsha

    diff_text = repo.git.diff(
        "--unified=0",
        "--no-ext-diff",
        "--find-renames",
        f"{stable_sha}..{target_sha}",
        "--",
    )
    diff_files = _parse_target_diff(diff_text)

    symbol_targets: dict[str, list[SymbolLocation]] = {}
    for changed_file in diff_files:
        if changed_file.status == "deleted":
            continue
        ext = changed_file.path.rsplit(".", 1)[-1].lower()
        language = EXTENSION_TO_LANGUAGE.get(f".{ext}")
        if not language:
            continue
        parser = get_parser(language)
        if not parser:
            continue
        try:
            source_bytes = target.tree / changed_file.path
            source_bytes = source_bytes.data_stream.read()
        except (KeyError, AttributeError):
            continue

        symbol_targets[changed_file.path] = get_symbol_locations(parser, language, source_bytes)

    matches = cross_reference_symbol_targets(diff_files, symbol_targets)
    return DeploymentDiffReport(target_sha, stable_sha, diff_files, matches)


def get_file_change_counts(
    repo_path: str,
    days: int = 90,
    max_commits: int = 1000,
) -> dict[str, int]:
    """
    Returns a dict of { repo-relative file path: commit_count }
    for all files touched within the last `days` days.

    Uses a single `git log --name-only` pass capped at `max_commits`
    instead of per-commit `commit.stats` subprocess calls.
    Returns an empty dict if git is unavailable or repo has no history.
    """
    if not GIT_AVAILABLE:
        return {}

    try:
        repo = Repo(repo_path)
    except (InvalidGitRepositoryError, Exception):
        return {}

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    counts: dict[str, int] = {}

    try:
        raw_log = repo.git.log(
            f"-n{max_commits}",
            "--pretty=format:__COMMIT__%ct",
            "--name-only",
        )
        seen_in_commit: set[str] = set()
        for raw_line in raw_log.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("__COMMIT__"):
                seen_in_commit.clear()
                ts_str = line[len("__COMMIT__"):].strip()
                commit_dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                if commit_dt < cutoff:
                    break
                continue
            if line not in seen_in_commit:
                seen_in_commit.add(line)
                counts[line] = counts.get(line, 0) + 1
    except Exception:
        pass

    return counts


def get_file_last_modified(
    repo_path: str,
    max_commits: int = 500,
) -> dict[str, str]:
    """
    Returns { repo-relative path: ISO date string of last commit }.
    Uses a single `git log --name-only` pass for fast startup.
    """
    if not GIT_AVAILABLE:
        return {}

    try:
        repo = Repo(repo_path)
    except Exception:
        return {}

    last_modified: dict[str, str] = {}

    try:
        raw_log = repo.git.log(
            f"-n{max_commits}",
            "--pretty=format:__COMMIT__%ct",
            "--name-only",
        )
        current_dt = ""
        for raw_line in raw_log.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("__COMMIT__"):
                ts_str = line[len("__COMMIT__"):].strip()
                current_dt = datetime.fromtimestamp(int(ts_str)).strftime("%Y-%m-%d")
                continue
            if current_dt and line not in last_modified:
                last_modified[line] = current_dt
    except Exception:
        pass

    return last_modified


def get_recent_commits_for_files(
    repo_path: str,
    file_paths: Iterable[str],
    *,
    max_commits: int = 200,
    max_per_file: int = 3,
) -> dict[str, list[dict[str, str]]]:
    """
    Returns { repo-relative path: [{"sha": ..., "author": ..., "date": ..., "message": ...}] }
    for the requested `file_paths` using a single `git log` pass.
    """
    target_set = {p for p in file_paths if p}
    if not GIT_AVAILABLE or not target_set:
        return {}

    try:
        repo = Repo(repo_path)
    except Exception:
        return {}

    commits_by_file: dict[str, list[dict[str, str]]] = {p: [] for p in target_set}
    try:
        raw_log = repo.git.log(
            f"-n{max_commits}",
            "--pretty=format:__COMMIT__%h|%an|%ad|%s",
            "--date=short",
            "--name-only",
        )
        current_meta: Optional[dict[str, str]] = None
        seen_in_commit: set[str] = set()
        for raw_line in raw_log.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("__COMMIT__"):
                seen_in_commit.clear()
                parts = line[len("__COMMIT__"):].split("|", 3)
                if len(parts) == 4:
                    current_meta = {
                        "sha": parts[0].strip(),
                        "author": parts[1].strip(),
                        "date": parts[2].strip(),
                        "message": parts[3].strip(),
                    }
                else:
                    current_meta = None
                continue
            if current_meta is not None and line in target_set and line not in seen_in_commit:
                seen_in_commit.add(line)
                if len(commits_by_file[line]) < max_per_file:
                    commits_by_file[line].append(dict(current_meta))
    except Exception:
        pass

    return {k: v for k, v in commits_by_file.items() if v}
