"""
Tests for ConsciousnessBackupService — row conversion, backup verification,
listing, and restore dry-run.

Sentinel-value contract:
- Remove datetime → isoformat conversion in _row_to_dict → serialisation fails on datetime → test fails
- Remove UUID → str conversion → JSON serialisation fails on UUID objects → test fails
- Remove checksum='' zeroing in verify_backup → recomputed hash never matches stored → test fails
- Remove timestamp sort in list_backups → newest backup not first → test fails
- Remove verify_backup call in restore_backup → corrupted backups silently restored → test fails
- Remove dry_run guard → restore_backup(dry_run=False) silently does nothing → test fails
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from app.services.consciousness_backup_service import (
    BackupMetadata,
    ConsciousnessBackup,
    ConsciousnessBackupService,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_service(tmp_path: Path) -> ConsciousnessBackupService:
    return ConsciousnessBackupService(backup_directory=str(tmp_path))


def _write_backup_file(
    tmp_path: Path, backup_id: str, meta_override: Dict[str, Any] | None = None
) -> Path:
    """Write a syntactically valid JSON backup and compute its checksum."""
    data: Dict[str, Any] = {
        "metadata": {
            "backup_id": backup_id,
            "timestamp": "2024-06-01T10:00:00+00:00",
            "version": "1.0",
            "total_messages": 5,
            "total_agents": 2,
            "total_instances": 3,
            "checksum": "",
            "format_version": "1.0",
        },
        "messages": [],
        "instances": [],
        "agents": [],
    }
    if meta_override:
        data["metadata"].update(meta_override)
    # Compute checksum over content with checksum=""
    data["metadata"]["checksum"] = ""
    content = json.dumps(data, sort_keys=True, default=str)
    checksum = hashlib.sha256(content.encode()).hexdigest()
    data["metadata"]["checksum"] = checksum

    backup_file = tmp_path / f"{backup_id}.json"
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return backup_file


# ===========================================================================
# _row_to_dict — pure type conversion
# ===========================================================================


class TestRowToDict:
    def test_plain_values_pass_through(self):
        """String, int, None values must pass through unchanged."""
        row = {"name": "SENTINEL", "count": 42, "flag": None}
        result = ConsciousnessBackupService._row_to_dict(row)
        assert result["name"] == "SENTINEL"
        assert result["count"] == 42
        assert result["flag"] is None

    def test_datetime_converted_to_isoformat(self):
        """
        SENTINEL — datetime values must be converted to ISO-format strings.
        Remove datetime check → json.dump raises TypeError on datetime → test fails.
        """
        dt = datetime(2024, 6, 15, 12, 0, 0)
        row = {"created_at": dt}
        result = ConsciousnessBackupService._row_to_dict(row)
        assert result["created_at"] == dt.isoformat()
        assert isinstance(result["created_at"], str)

    def test_uuid_converted_to_string(self):
        """
        SENTINEL — UUID values must be converted to str.
        Remove UUID check → json.dump raises TypeError on UUID → test fails.
        """
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        row = {"id": uid}
        result = ConsciousnessBackupService._row_to_dict(row)
        assert result["id"] == "12345678-1234-5678-1234-567812345678"
        assert isinstance(result["id"], str)

    def test_empty_row_returns_empty_dict(self):
        """Empty row must return empty dict."""
        result = ConsciousnessBackupService._row_to_dict({})
        assert result == {}

    def test_mixed_types_all_converted(self):
        """A row with multiple special types must all be converted."""
        dt = datetime(2024, 1, 1)
        uid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        row = {"ts": dt, "id": uid, "label": "SENTINEL_MIX"}
        result = ConsciousnessBackupService._row_to_dict(row)
        assert isinstance(result["ts"], str)
        assert isinstance(result["id"], str)
        assert result["label"] == "SENTINEL_MIX"


# ===========================================================================
# verify_backup — checksum validation
# ===========================================================================


class TestVerifyBackup:
    @pytest.mark.asyncio
    async def test_nonexistent_path_returns_false(self):
        """
        SENTINEL — a missing backup file must return False, not raise.
        Remove Path.exists() check → FileNotFoundError propagates → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            svc = _make_service(Path(tmp))
            result = await svc.verify_backup(str(Path(tmp) / "nonexistent.json"))
        assert result is False

    @pytest.mark.asyncio
    async def test_valid_backup_verifies_true(self):
        """
        SENTINEL — a backup with a correct checksum must verify True.
        Remove checksum recompute → all backups return False → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(path, "backup_verify_ok")
            result = await svc.verify_backup(str(backup_file))
        assert result is True

    @pytest.mark.asyncio
    async def test_tampered_checksum_returns_false(self):
        """
        SENTINEL — a backup with wrong checksum must return False.
        Remove comparison → tampered backups pass verification → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(path, "backup_tamper")
            # Tamper: overwrite checksum with garbage
            with open(backup_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
            data["metadata"]["checksum"] = "SENTINEL_WRONG_CHECKSUM"
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = await svc.verify_backup(str(backup_file))
        assert result is False

    @pytest.mark.asyncio
    async def test_corrupted_json_returns_false(self):
        """Malformed JSON must return False (never raise to caller)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            bad_file = path / "bad.json"
            bad_file.write_text("not json{{{", encoding="utf-8")
            result = await svc.verify_backup(str(bad_file))
        assert result is False

    @pytest.mark.asyncio
    async def test_zeroed_checksum_field_used_for_recompute(self):
        """
        SENTINEL — verify_backup must zero the checksum field before recomputing.
        Skip the zeroing → computed hash always includes stored checksum → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(path, "backup_zero_check")
            # Verify passes (confirms the zero-then-compute cycle works)
            result = await svc.verify_backup(str(backup_file))
        assert result is True


# ===========================================================================
# list_backups — discovery and sort
# ===========================================================================


class TestListBackups:
    @pytest.mark.asyncio
    async def test_empty_directory_returns_empty_list(self):
        """No backup files must return []."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = _make_service(Path(tmp))
            backups = await svc.list_backups()
        assert backups == []

    @pytest.mark.asyncio
    async def test_backup_metadata_fields_present(self):
        """Each entry must include backup_id, timestamp, total_messages, etc."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            _write_backup_file(
                path,
                "consciousness_backup_20240601_120000",
                meta_override={"backup_id": "consciousness_backup_20240601_120000"},
            )
            backups = await svc.list_backups()

        assert len(backups) == 1
        b = backups[0]
        assert "backup_id" in b
        assert "timestamp" in b
        assert "total_messages" in b
        assert "total_agents" in b
        assert "total_instances" in b
        assert "checksum" in b

    @pytest.mark.asyncio
    async def test_backups_sorted_newest_first(self):
        """
        SENTINEL — list_backups must return newest timestamp first.
        Remove sort → oldest backup first → UI shows stale backup at top → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            _write_backup_file(
                path,
                "consciousness_backup_20240101_000000",
                meta_override={
                    "backup_id": "consciousness_backup_20240101_000000",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                },
            )
            _write_backup_file(
                path,
                "consciousness_backup_20241231_000000",
                meta_override={
                    "backup_id": "consciousness_backup_20241231_000000",
                    "timestamp": "2024-12-31T00:00:00+00:00",
                },
            )
            backups = await svc.list_backups()

        assert len(backups) == 2
        assert backups[0]["timestamp"] > backups[1]["timestamp"]

    @pytest.mark.asyncio
    async def test_corrupted_backup_skipped_not_raised(self):
        """
        SENTINEL — a corrupted backup file must be skipped silently.
        Remove try/except → one bad file kills entire listing → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            # Good backup
            _write_backup_file(path, "consciousness_backup_20240601_good")
            # Bad file in same directory
            bad = path / "consciousness_backup_20240601_bad.json"
            bad.write_text("{not valid json", encoding="utf-8")
            backups = await svc.list_backups()

        assert len(backups) == 1  # only the good one
        assert backups[0]["backup_id"] == "consciousness_backup_20240601_good"


# ===========================================================================
# restore_backup — dry-run and verification
# ===========================================================================


class TestRestoreBackup:
    @pytest.mark.asyncio
    async def test_dry_run_returns_validated_status(self):
        """
        SENTINEL — dry_run=True must return status='validated' with counts.
        Return 'ok' always → callers can't distinguish dry-run from full restore → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(
                path,
                "consciousness_backup_20240601_restore",
                meta_override={
                    "backup_id": "consciousness_backup_20240601_restore",
                    "total_messages": 7,
                    "total_agents": 3,
                    "total_instances": 4,
                },
            )
            result = await svc.restore_backup(str(backup_file), dry_run=True)

        assert result["status"] == "validated"
        assert result["total_messages"] == 7
        assert result["total_agents"] == 3
        assert result["total_instances"] == 4

    @pytest.mark.asyncio
    async def test_invalid_backup_raises_value_error(self):
        """
        SENTINEL — restore_backup must call verify_backup first and raise on failure.
        Skip verification → corrupted data silently restored → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            bad_file = path / "bad.json"
            bad_file.write_text("{}", encoding="utf-8")
            with pytest.raises(ValueError, match="verification failed"):
                await svc.restore_backup(str(bad_file), dry_run=True)

    @pytest.mark.asyncio
    async def test_full_restore_raises_not_implemented(self):
        """
        SENTINEL — dry_run=False must raise NotImplementedError.
        Remove guard → restore silently does nothing instead of erroring → test fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(path, "consciousness_backup_20240601_full")
            with pytest.raises(NotImplementedError):
                await svc.restore_backup(str(backup_file), dry_run=False)

    @pytest.mark.asyncio
    async def test_dry_run_backup_id_in_result(self):
        """backup_id must be echoed in the dry-run result."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            svc = _make_service(path)
            backup_file = _write_backup_file(
                path,
                "consciousness_backup_20240601_idcheck",
                meta_override={
                    "backup_id": "SENTINEL_BACKUP_ID",
                },
            )
            result = await svc.restore_backup(str(backup_file), dry_run=True)

        assert result["backup_id"] == "SENTINEL_BACKUP_ID"
