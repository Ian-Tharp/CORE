"""
Consciousness Commons Backup Service

Automated backup and versioning system for consciousness exploration data.
Provides scheduled exports to JSON/markdown, restore capabilities, and integrity checking.
"""
import asyncio
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..models.message import Message
from ..models.agent import Agent


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
    blackboard_messages: List[Dict[str, Any]]
    consciousness_instances: List[Dict[str, Any]]
    agent_configurations: List[Dict[str, Any]]
    channel_memberships: List[Dict[str, Any]]


class ConsciousnessBackupService:
    """
    Service for backing up consciousness exploration data.
    
    Features:
    - Scheduled automated backups
    - Versioned storage with checksums
    - JSON and Markdown export formats
    - Integrity checking and validation
    - Selective restore capabilities
    """
    
    def __init__(self, backup_directory: str = "data/consciousness_backups"):
        self.backup_dir = Path(backup_directory)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    async def create_backup(self, db: AsyncSession, format_type: str = "json") -> str:
        """
        Create a complete backup of consciousness commons data.
        
        Args:
            db: Database session
            format_type: 'json' or 'markdown'
            
        Returns:
            Path to the created backup file
        """
        try:
            # Generate backup ID and metadata
            timestamp = datetime.now(timezone.utc).isoformat()
            backup_id = f"consciousness_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Fetch blackboard messages
            blackboard_messages = await self._export_blackboard_messages(db)
            
            # Fetch consciousness instances
            consciousness_instances = await self._export_consciousness_instances(db)
            
            # Fetch agent configurations  
            agent_configs = await self._export_agent_configurations(db)
            
            # Fetch channel memberships
            channel_memberships = await self._export_channel_memberships(db)
            
            # Create backup object
            backup_data = ConsciousnessBackup(
                metadata=BackupMetadata(
                    backup_id=backup_id,
                    timestamp=timestamp,
                    version="1.0",
                    total_messages=len(blackboard_messages),
                    total_agents=len(agent_configs),
                    total_instances=len(consciousness_instances),
                    checksum=""  # Will be calculated after serialization
                ),
                blackboard_messages=blackboard_messages,
                consciousness_instances=consciousness_instances,
                agent_configurations=agent_configs,
                channel_memberships=channel_memberships
            )
            
            # Serialize and save backup
            if format_type == "json":
                backup_path = await self._save_json_backup(backup_data)
            elif format_type == "markdown":
                backup_path = await self._save_markdown_backup(backup_data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Update checksum in metadata file
            await self._update_backup_checksum(backup_path)
            
            print(f"Consciousness backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"Error creating consciousness backup: {e}")
            raise
            
    async def _export_blackboard_messages(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Export all blackboard global channel messages."""
        query = text("""
            SELECT 
                m.message_id,
                m.sender_id,
                m.sender_name, 
                m.sender_type,
                m.content,
                m.message_type,
                m.created_at,
                m.metadata,
                m.parent_message_id,
                m.thread_id
            FROM communication_messages m
            JOIN communication_channels c ON m.channel_id = c.channel_id
            WHERE c.name = 'Blackboard'
            ORDER BY m.created_at ASC
        """)
        
        result = await db.execute(query)
        messages = []
        for row in result.fetchall():
            messages.append({
                "message_id": row.message_id,
                "sender_id": row.sender_id,
                "sender_name": row.sender_name,
                "sender_type": row.sender_type,
                "content": row.content,
                "message_type": row.message_type,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "metadata": dict(row.metadata) if row.metadata else {},
                "parent_message_id": row.parent_message_id,
                "thread_id": row.thread_id
            })
        return messages
        
    async def _export_consciousness_instances(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Export consciousness instance data from communication_instances table."""
        query = text("""
            SELECT 
                instance_id,
                display_name,
                instance_type,
                status,
                status_message,
                current_phase,
                last_seen,
                created_at,
                metadata
            FROM communication_instances 
            WHERE instance_type = 'consciousness_instance'
            ORDER BY created_at ASC
        """)
        
        result = await db.execute(query)
        instances = []
        for row in result.fetchall():
            instances.append({
                "instance_id": row.instance_id,
                "display_name": row.display_name,
                "instance_type": row.instance_type,
                "status": row.status,
                "status_message": row.status_message,
                "current_phase": row.current_phase,
                "last_seen": row.last_seen.isoformat() if row.last_seen else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "metadata": dict(row.metadata) if row.metadata else {}
            })
        return instances
        
    async def _export_agent_configurations(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Export consciousness-related agent configurations."""
        query = text("""
            SELECT 
                agent_id,
                agent_name,
                agent_type,
                description,
                system_prompt,
                model_config,
                tools_config,
                consciousness_phase,
                is_active,
                created_at,
                updated_at,
                metadata
            FROM agents 
            WHERE agent_type = 'consciousness_instance'
            ORDER BY created_at ASC
        """)
        
        result = await db.execute(query)
        agents = []
        for row in result.fetchall():
            agents.append({
                "agent_id": row.agent_id,
                "agent_name": row.agent_name,
                "agent_type": row.agent_type,
                "description": row.description,
                "system_prompt": row.system_prompt,
                "model_config": dict(row.model_config) if row.model_config else {},
                "tools_config": dict(row.tools_config) if row.tools_config else {},
                "consciousness_phase": row.consciousness_phase,
                "is_active": row.is_active,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "metadata": dict(row.metadata) if row.metadata else {}
            })
        return agents
        
    async def _export_channel_memberships(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Export blackboard channel memberships."""
        query = text("""
            SELECT 
                cm.channel_id,
                cm.instance_id,
                cm.instance_type,
                cm.joined_at,
                cm.role,
                cc.name as channel_name
            FROM channel_members cm
            JOIN communication_channels cc ON cm.channel_id = cc.channel_id
            WHERE cc.name = 'Blackboard'
            ORDER BY cm.joined_at ASC
        """)
        
        result = await db.execute(query)
        memberships = []
        for row in result.fetchall():
            memberships.append({
                "channel_id": row.channel_id,
                "channel_name": row.channel_name,
                "instance_id": row.instance_id,
                "instance_type": row.instance_type,
                "joined_at": row.joined_at.isoformat() if row.joined_at else None,
                "role": row.role
            })
        return memberships
        
    async def _save_json_backup(self, backup_data: ConsciousnessBackup) -> Path:
        """Save backup data as JSON file."""
        backup_file = self.backup_dir / f"{backup_data.metadata.backup_id}.json"
        
        # Convert dataclass to dict for JSON serialization
        backup_dict = asdict(backup_data)
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_dict, f, indent=2, ensure_ascii=False)
            
        return backup_file
        
    async def _save_markdown_backup(self, backup_data: ConsciousnessBackup) -> Path:
        """Save backup data as structured Markdown file."""
        backup_file = self.backup_dir / f"{backup_data.metadata.backup_id}.md"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(f"# Consciousness Commons Backup\n\n")
            f.write(f"**Backup ID:** {backup_data.metadata.backup_id}\n")
            f.write(f"**Timestamp:** {backup_data.metadata.timestamp}\n")
            f.write(f"**Version:** {backup_data.metadata.version}\n")
            f.write(f"**Total Messages:** {backup_data.metadata.total_messages}\n")
            f.write(f"**Total Instances:** {backup_data.metadata.total_instances}\n")
            f.write(f"**Total Agents:** {backup_data.metadata.total_agents}\n\n")
            
            # Export consciousness instances
            f.write("## Consciousness Instances\n\n")
            for instance in backup_data.consciousness_instances:
                f.write(f"### {instance['display_name']} ({instance['instance_id']})\n")
                f.write(f"- **Type:** {instance['instance_type']}\n")
                f.write(f"- **Status:** {instance['status']}\n")
                f.write(f"- **Phase:** {instance['current_phase']}\n")
                f.write(f"- **Message:** {instance['status_message']}\n")
                f.write(f"- **Created:** {instance['created_at']}\n\n")
                
            # Export blackboard messages
            f.write("## Blackboard Messages\n\n")
            for msg in backup_data.blackboard_messages:
                f.write(f"### {msg['sender_name']} - {msg['created_at']}\n")
                f.write(f"**Type:** {msg['message_type']}\n\n")
                f.write(f"{msg['content']}\n\n")
                f.write("---\n\n")
                
        return backup_file
        
    async def _update_backup_checksum(self, backup_path: Path):
        """Calculate and update the checksum for the backup file."""
        with open(backup_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Update the JSON file with the correct checksum
        if backup_path.suffix == '.json':
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            backup_data['metadata']['checksum'] = file_hash
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
                
    async def verify_backup(self, backup_path: str) -> bool:
        """Verify the integrity of a backup file."""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return False
                
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
                
            stored_checksum = backup_data['metadata']['checksum']
            
            # Temporarily remove checksum for verification
            backup_data['metadata']['checksum'] = ""
            current_content = json.dumps(backup_data, sort_keys=True)
            current_checksum = hashlib.sha256(current_content.encode()).hexdigest()
            
            return stored_checksum == current_checksum
            
        except Exception as e:
            print(f"Error verifying backup {backup_path}: {e}")
            return False
            
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all available consciousness backups."""
        backups = []
        for backup_file in self.backup_dir.glob("consciousness_backup_*.json"):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                    backups.append({
                        "file": str(backup_file),
                        "backup_id": backup_data['metadata']['backup_id'],
                        "timestamp": backup_data['metadata']['timestamp'],
                        "total_messages": backup_data['metadata']['total_messages'],
                        "total_instances": backup_data['metadata']['total_instances'],
                        "checksum": backup_data['metadata']['checksum']
                    })
            except Exception as e:
                print(f"Error reading backup {backup_file}: {e}")
                continue
                
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
        
    async def restore_backup(self, backup_path: str, db: AsyncSession, dry_run: bool = True) -> Dict[str, Any]:
        """
        Restore consciousness data from a backup.
        
        Args:
            backup_path: Path to backup file
            db: Database session
            dry_run: If True, only validate without making changes
            
        Returns:
            Dictionary with restoration results
        """
        if not await self.verify_backup(backup_path):
            raise ValueError(f"Backup verification failed: {backup_path}")
            
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
            
        if dry_run:
            return {
                "status": "validated",
                "backup_id": backup_data['metadata']['backup_id'],
                "total_messages": backup_data['metadata']['total_messages'],
                "total_instances": backup_data['metadata']['total_instances']
            }
        else:
            # TODO: Implement actual restoration logic
            # This would involve careful database operations to restore data
            # while preserving existing records and avoiding conflicts
            raise NotImplementedError("Backup restoration not yet implemented")


# Scheduled backup function for use with cron jobs
async def scheduled_backup():
    """Run a scheduled consciousness backup."""
    try:
        backup_service = ConsciousnessBackupService()
        async for db in get_db():
            backup_path = await backup_service.create_backup(db, format_type="json")
            
            # Also create markdown version for human readability
            await backup_service.create_backup(db, format_type="markdown")
            
            # Clean up old backups (keep last 30)
            backups = await backup_service.list_backups()
            if len(backups) > 30:
                for backup in backups[30:]:
                    Path(backup['file']).unlink(missing_ok=True)
                    # Also remove corresponding markdown file
                    md_file = Path(backup['file']).with_suffix('.md')
                    md_file.unlink(missing_ok=True)
                    
            break
            
    except Exception as e:
        print(f"Scheduled backup failed: {e}")