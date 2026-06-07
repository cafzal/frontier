"""JSON file store for problems. One file per problem in ./data/."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .models import Problem

# store.py is at <repo-root>/engine/store.py, so two parents up is the repo root.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Store:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, problem_id: str) -> Path:
        return self.data_dir / f"{_safe_id(problem_id, 'problem_id')}.json"

    def save(self, problem: Problem) -> None:
        path = self._path(problem.problem_id)
        _atomic_write(path, problem.model_dump_json(indent=2))

    def load(self, problem_id: str) -> Problem:
        path = self._path(problem_id)
        if not path.exists():
            raise FileNotFoundError(f"Problem {problem_id} not found")
        return Problem.model_validate_json(path.read_text())

    def list(self) -> list[dict]:
        problems = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                p = Problem.model_validate_json(path.read_text())
                problems.append({
                    "problem_id": p.problem_id,
                    "name": p.name,
                    "domain": p.domain,
                    "created_at": p.created_at.isoformat(),
                    "updated_at": p.updated_at.isoformat(),
                    "status": {
                        "objectives": len(p.objectives),
                        "options": len(p.options),
                        "scores_complete": _scores_completeness(p),
                        "has_run": p.run is not None,
                        "results_stale": p.results_stale,
                        "total_runs": len(p.runs) + (1 if p.run else 0),
                    },
                })
            except Exception:
                continue
        return problems

    def delete(self, problem_id: str) -> None:
        path = self._path(problem_id)
        if not path.exists():
            raise FileNotFoundError(f"Problem {problem_id} not found")
        path.unlink()

    def exists(self, problem_id: str) -> bool:
        return self._path(problem_id).exists()

    def run_result_path(self, problem_id: str, run_id: str, scenario: str | None = None) -> Path:
        """Path where a solve run's full result is (or will be) persisted.

        Layout: ``<data_dir>/runs/<problem_id>/[<scenario>/]<run_id>.json``.
        Scenario runs are nested one level deeper, one file per scenario.
        """
        runs_dir = self.data_dir / "runs" / _safe_id(problem_id, "problem_id")
        if scenario:
            runs_dir = runs_dir / _safe_slug(scenario)
        return runs_dir / f"{_safe_id(run_id, 'run_id')}.json"

    def write_run_result(self, problem_id: str, run_id: str, payload: dict, scenario: str | None = None) -> Path:
        """Write a solve run's full result to disk and return the path."""
        path = self.run_result_path(problem_id, run_id, scenario=scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(payload, indent=2))
        return path


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Writes a uniquely-named sibling temp file, then ``os.replace`` (an atomic
    same-directory rename). A concurrent reader therefore never sees a
    half-written file, and concurrent writers — e.g. a background solve worker
    saving a run while a foreground ``model/update`` saves an edit — each get
    their own temp file, so the only race left is which complete write wins the
    rename (resolved at the engine layer by the solve worker's stale-guard).
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_id(value: str, label: str) -> str:
    """Reject ids that could escape the data dir via path traversal.

    ``problem_id`` / ``run_id`` arrive from MCP clients; without this an id like
    ``../../etc/x`` would let load/save/delete operate outside ``data_dir``.
    Legit ids are UUIDs (hex + hyphens), which pass.
    """
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {label}: must be alphanumeric, '-' or '_'")
    return value


def _safe_slug(name: str) -> str:
    """Filesystem-safe slug from an arbitrary string."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _scores_completeness(p: Problem) -> float:
    total = len(p.objectives) * len(p.options)
    if total == 0:
        return 0.0
    return len(p.scores) / total
