"""Crash-resistant, append-only experiment tracking and result export."""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from prefpo.lattice.config import ExperimentConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert supported dataclasses/Pydantic models before strict JSON encoding."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Tracked JSON mappings must have string keys")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported experiment tracking value: {type(value).__name__}")


class ExperimentTracker:
    """Write configuration snapshots, event logs, policy versions, and metrics."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        base_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        root = Path(base_dir or config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{_slug(config.name)}-{timestamp}-{uuid.uuid4().hex[:8]}"
        if Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a single safe path component")
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=False, exist_ok=False)
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._events_path = self.run_dir / "events.jsonl"
        self._write_json("config.json", config.model_dump(mode="json"))
        self.log_event("run_started", {"name": config.name, "seed": config.seed})

    def _write_json(self, filename: str, payload: Any) -> Path:
        path = self.run_dir / filename
        temp_path = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_safe)
        temp_path.write_text(text + "\n", encoding="utf-8")
        os.replace(temp_path, path)
        return path

    def log_event(self, event_type: str, payload: Any) -> None:
        """Append one sequenced event; reject writes after run completion."""
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        serialized = json.loads(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_safe)
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot append to a completed experiment run")
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp": _utc_now(),
                "event_type": event_type,
                "payload": serialized,
            }
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with self._events_path.open("a", encoding="utf-8") as file:
                file.write(encoded + "\n")
                file.flush()
                os.fsync(file.fileno())

    def log_policy(self, phase: str, policy: Any, *, metrics: Any = None) -> Path:
        """Persist a versioned policy snapshot and emit a compact event."""
        policy_dict = _json_safe(policy)
        policies_dir = self.run_dir / "policies"
        policies_dir.mkdir(exist_ok=True)
        fingerprint = str(policy_dict.get("fingerprint", uuid.uuid4().hex[:8]))
        filename = f"{_slug(phase)}-{fingerprint[:16]}.json"
        path = policies_dir / filename
        temp_path = path.with_suffix(".json.tmp")
        payload = {
            "phase": phase,
            "recorded_at": _utc_now(),
            "policy": policy_dict,
            "metrics": _json_safe(metrics) if metrics is not None else None,
        }
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_safe)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
        self.log_event("policy_snapshot", {"phase": phase, "file": str(path.relative_to(self.run_dir))})
        return path

    def log_metrics(self, phase: str, metrics: Any) -> Path:
        payload = _json_safe(metrics)
        filename = f"{_slug(phase)}-metrics.json"
        path = self._write_json(filename, {"phase": phase, "recorded_at": _utc_now(), **payload})
        self.log_event("metrics", {"phase": phase, "file": filename})
        return path

    def export_prediction_rows(
        self,
        filename: str,
        rows: Iterable[dict[str, Any]],
    ) -> Path:
        """Export flat evaluation rows in CSV for external analysis."""
        if Path(filename).name != filename or not filename.endswith(".csv"):
            raise ValueError("filename must be a simple .csv filename")
        values = list(rows)
        if not values:
            raise ValueError("Cannot export an empty prediction row set")
        columns = list(dict.fromkeys(key for row in values for key in row))
        path = self.run_dir / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(values)
        self.log_event("prediction_export", {"file": filename, "rows": len(values)})
        return path

    def complete(self, summary: dict[str, Any]) -> Path:
        """Write immutable completion summary and close the append-only event stream."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Experiment run is already completed")
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp": _utc_now(),
                "event_type": "run_completed",
                "payload": json.loads(
                    json.dumps(summary, ensure_ascii=False, allow_nan=False, default=_json_safe)
                ),
            }
            with self._events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            path = self._write_json("summary.json", summary)
            self._closed = True
            return path


def _slug(value: str) -> str:
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    if not cleaned:
        raise ValueError("Name must contain at least one alphanumeric character")
    return cleaned[:80]
