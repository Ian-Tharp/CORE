"""
Consciousness Commons Backup Service

Automated backup and versioning system for consciousness exploration data.
Provides scheduled exports to JSON/markdown, restore capabilities, and integrity checking.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a consciousness backup."""

    backup_id: str
    timestamp: str
    version: str
    total_messages: int
    total_agents: int
    total_instances: int
    checksum: str
    format_version: str = "1.0"


@dataclass
class ConsciousnessBackup:
    """Complete consciousness commons backup data structure."""

    metadata: BackupMetadata
    messages: List[Dict[str, Any]]
    instances: List[Dict[str, Any]]
    agents: List[Dict[str, Any]]


class ConsciousnessBackupService:
    """
    Service for backing up consciousness exploration data.

    Uses asyncpg directly via get_db_pool() — no SQLAlchemy.
    """

    def __init__(self, backup_directory: str = "data/consciousness_backups"):
        self.backup_dir = Path(backup_directory)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    async def create_backup(self, format_type: str = "json") -> str:
        """
        Create a complete backup of consciousness commons data.

        Args:
            format_type: 'json' or 'markdown'

        Returns:
            Path to the created backup file
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        backup_id = f"consciousness_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            agents = await self._export_agents(conn)
            instances = await self._export_instances(conn)
            messages = await self._export_messages(conn)

        backup_data = ConsciousnessBackup(
            metadata=BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                version="1.0",
                total_messages=len(messages),
                total_agents=len(agents),
                total_instances=len(instances),
                checksum="",
            ),
            messages=messages,
            instances=instances,
            agents=agents,
        )

        if format_type == "json":
            backup_path = await self._save_json_backup(backup_data)
        elif format_type == "markdown":
            backup_path = await self._save_markdown_backup(backup_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        await self._update_backup_checksum(backup_path)
        logger.info(f"Consciousness backup created: {backup_path}")
        return str(backup_path)

    # ------------------------------------------------------------------
    # Export helpers (raw asyncpg)
    # ------------------------------------------------------------------

    async def _export_agents(self, conn) -> List[Dict[str, Any]]:
        """Export all agent configurations."""
        rows = await conn.fetch("SELECT * FROM agents ORDER BY created_at ASC")
        return [self._row_to_dict(r) for r in rows]

    async def _export_instances(self, conn) -> List[Dict[str, Any]]:
        """Export all agent instances."""
        rows = await conn.fetch("SELECT * FROM agent_instances ORDER BY created_at ASC")
        return [self._row_to_dict(r) for r in rows]

    async def _export_messages(self, conn) -> List[Dict[str, Any]]:
        """Export communication messages."""
        rows = await conn.fetch(
            "SELECT * FROM communication_messages ORDER BY created_at ASC"
        )
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Convert an asyncpg Record to a JSON-safe dict."""
        d: Dict[str, Any] = {}
        for key, value in dict(row).items():
            if isinstance(value, datetime):
                d[key] = value.isoformat()
            elif hasattr(value, "__str__") and type(value).__name__ == "UUID":
                d[key] = str(value)
            else:
                d[key] = value
        return d

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _save_json_backup(self, backup_data: ConsciousnessBackup) -> Path:
        backup_file = self.backup_dir / f"{backup_data.metadata.backup_id}.json"
        backup_dict = asdict(backup_data)
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_dict, f, indent=2, ensure_ascii=False, default=str)
        return backup_file

    async def _save_markdown_backup(self, backup_data: ConsciousnessBackup) -> Path:
        backup_file = self.backup_dir / f"{backup_data.metadata.backup_id}.md"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write(f"# Consciousness Commons Backup\n\n")
            f.write(f"**Backup ID:** {backup_data.metadata.backup_id}\n")
            f.write(f"**Timestamp:** {backup_data.metadata.timestamp}\n")
            f.write(f"**Agents:** {backup_data.metadata.total_agents}\n")
            f.write(f"**Instances:** {backup_data.metadata.total_instances}\n")
            f.write(f"**Messages:** {backup_data.metadata.total_messages}\n\n")

            f.write("## Agents\n\n")
            for agent in backup_data.agents:
                f.write(
                    f"### {agent.get('agent_name', agent.get('agent_id', 'unknown'))}\n"
                )
                for k, v in agent.items():
                    f.write(f"- **{k}:** {v}\n")
                f.write("\n")

            f.write("## Instances\n\n")
            for inst in backup_data.instances:
                f.write(f"### {inst.get('agent_id', inst.get('id', 'unknown'))}\n")
                for k, v in inst.items():
                    f.write(f"- **{k}:** {v}\n")
                f.write("\n")

            f.write("## Messages\n\n")
            for msg in backup_data.messages:
                sender = msg.get("sender_name", msg.get("sender_id", "unknown"))
                f.write(f"**{sender}** ({msg.get('created_at', '')}):\n")
                f.write(f"{msg.get('content', '')}\n\n---\n\n")

        return backup_file

    async def _update_backup_checksum(self, backup_path: Path):
        with open(backup_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if backup_path.suffix == ".json":
            with open(backup_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["metadata"]["checksum"] = file_hash
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Verification & listing
    # ------------------------------------------------------------------

    async def verify_backup(self, backup_path: str) -> bool:
        try:
            p = Path(backup_path)
            if not p.exists():
                return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored = data["metadata"]["checksum"]
            # Recompute on the file content before checksum was written
            data["metadata"]["checksum"] = ""
            content = json.dumps(data, sort_keys=True, default=str)
            computed = hashlib.sha256(content.encode()).hexdigest()
            return stored == computed
        except Exception as e:
            logger.error(f"Error verifying backup {backup_path}: {e}")
            return False

    async def list_backups(self) -> List[Dict[str, Any]]:
        backups = []
        for f in self.backup_dir.glob("consciousness_backup_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                meta = data["metadata"]
                backups.append(
                    {
                        "file": str(f),
                        "backup_id": meta["backup_id"],
                        "timestamp": meta["timestamp"],
                        "total_messages": meta["total_messages"],
                        "total_agents": meta["total_agents"],
                        "total_instances": meta["total_instances"],
                        "checksum": meta["checksum"],
                    }
                )
            except Exception as e:
                logger.warning(f"Error reading backup {f}: {e}")
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    async def restore_backup(
        self, backup_path: str, dry_run: bool = True
    ) -> Dict[str, Any]:
        if not await self.verify_backup(backup_path):
            raise ValueError(f"Backup verification failed: {backup_path}")

        with open(backup_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if dry_run:
            return {
                "status": "validated",
                "backup_id": data["metadata"]["backup_id"],
                "total_messages": data["metadata"]["total_messages"],
                "total_agents": data["metadata"]["total_agents"],
                "total_instances": data["metadata"]["total_instances"],
            }
        else:
            raise NotImplementedError("Full restoration not yet implemented")


async def scheduled_backup():
    """Run a scheduled consciousness backup."""
    try:
        svc = ConsciousnessBackupService()
        await svc.create_backup(format_type="json")
        await svc.create_backup(format_type="markdown")

        backups = await svc.list_backups()
        if len(backups) > 30:
            for b in backups[30:]:
                Path(b["file"]).unlink(missing_ok=True)
                Path(b["file"]).with_suffix(".md").unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Scheduled backup failed: {e}")
