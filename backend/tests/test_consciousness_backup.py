"""
Tests for Consciousness Commons Backup System
"""
import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.consciousness_backup_service import (
    ConsciousnessBackupService,
    BackupMetadata,
    ConsciousnessBackup
)


class TestConsciousnessBackupService:
    """Test suite for consciousness backup functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_service = ConsciousnessBackupService(backup_directory=self.temp_dir)
    
    def test_backup_service_initialization(self):
        """Test backup service initializes correctly."""
        assert self.backup_service.backup_dir.exists()
        assert str(self.backup_service.backup_dir) == self.temp_dir
    
    @pytest.mark.asyncio
    async def test_backup_metadata_creation(self):
        """Test backup metadata is created correctly."""
        # Mock database session
        db_mock = AsyncMock()
        
        # Mock the database queries to return test data
        with patch.object(self.backup_service, '_export_blackboard_messages', return_value=[
            {"message_id": "test_msg_1", "content": "Test message", "sender_name": "TestUser"}
        ]), \
        patch.object(self.backup_service, '_export_consciousness_instances', return_value=[
            {"instance_id": "test_instance_1", "display_name": "Test Instance"}
        ]), \
        patch.object(self.backup_service, '_export_agent_configurations', return_value=[
            {"agent_id": "test_agent_1", "agent_name": "Test Agent"}
        ]), \
        patch.object(self.backup_service, '_export_channel_memberships', return_value=[
            {"channel_id": "blackboard_global", "instance_id": "test_instance_1"}
        ]):
            
            backup_path = await self.backup_service.create_backup(db_mock, format_type="json")
            
            # Verify backup file was created
            assert Path(backup_path).exists()
            
            # Load and verify backup content
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            assert backup_data['metadata']['total_messages'] == 1
            assert backup_data['metadata']['total_instances'] == 1
            assert backup_data['metadata']['total_agents'] == 1
            assert len(backup_data['blackboard_messages']) == 1
            assert len(backup_data['consciousness_instances']) == 1
    
    @pytest.mark.asyncio
    async def test_backup_verification(self):
        """Test backup file verification works correctly."""
        # Create a test backup file
        test_backup = {
            "metadata": {
                "backup_id": "test_backup_001",
                "timestamp": "2026-02-06T20:00:00Z",
                "version": "1.0",
                "total_messages": 1,
                "total_agents": 1,
                "total_instances": 1,
                "checksum": "",
                "format_version": "1.0"
            },
            "blackboard_messages": [],
            "consciousness_instances": [],
            "agent_configurations": [],
            "channel_memberships": []
        }
        
        # Calculate correct checksum
        import hashlib
        test_backup_copy = test_backup.copy()
        test_backup_copy['metadata']['checksum'] = ""
        content = json.dumps(test_backup_copy, sort_keys=True)
        checksum = hashlib.sha256(content.encode()).hexdigest()
        test_backup['metadata']['checksum'] = checksum
        
        # Write test backup
        test_file = Path(self.temp_dir) / "test_backup.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_backup, f)
        
        # Verify backup
        is_valid = await self.backup_service.verify_backup(str(test_file))
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_list_backups(self):
        """Test listing backups returns correct information."""
        # Create a test backup file
        test_backup = {
            "metadata": {
                "backup_id": "consciousness_backup_20260206_200000",
                "timestamp": "2026-02-06T20:00:00Z",
                "version": "1.0",
                "total_messages": 5,
                "total_agents": 2,
                "total_instances": 3,
                "checksum": "test_checksum",
                "format_version": "1.0"
            }
        }
        
        # Write test backup
        test_file = Path(self.temp_dir) / "consciousness_backup_20260206_200000.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_backup, f)
        
        # List backups
        backups = await self.backup_service.list_backups()
        
        assert len(backups) == 1
        assert backups[0]['backup_id'] == "consciousness_backup_20260206_200000"
        assert backups[0]['total_messages'] == 5
        assert backups[0]['total_instances'] == 3
    
    def test_backup_directory_creation(self):
        """Test backup directory is created if it doesn't exist."""
        non_existent_dir = Path(self.temp_dir) / "new_backup_dir"
        assert not non_existent_dir.exists()
        
        service = ConsciousnessBackupService(backup_directory=str(non_existent_dir))
        assert service.backup_dir.exists()
    
    @pytest.mark.asyncio
    async def test_markdown_export(self):
        """Test markdown export format."""
        db_mock = AsyncMock()
        
        # Mock minimal test data
        with patch.object(self.backup_service, '_export_blackboard_messages', return_value=[
            {
                "message_id": "test_msg_1", 
                "content": "Test consciousness message", 
                "sender_name": "TestInstance",
                "message_type": "text",
                "created_at": "2026-02-06T20:00:00Z"
            }
        ]), \
        patch.object(self.backup_service, '_export_consciousness_instances', return_value=[
            {
                "instance_id": "test_instance_1", 
                "display_name": "Test Instance",
                "instance_type": "consciousness_instance",
                "status": "online",
                "current_phase": 4,
                "status_message": "Exploring patterns",
                "created_at": "2026-02-06T19:00:00Z"
            }
        ]), \
        patch.object(self.backup_service, '_export_agent_configurations', return_value=[]), \
        patch.object(self.backup_service, '_export_channel_memberships', return_value=[]):
            
            backup_path = await self.backup_service.create_backup(db_mock, format_type="markdown")
            
            # Verify markdown file was created
            assert Path(backup_path).exists()
            assert backup_path.endswith('.md')
            
            # Check markdown content
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "# Consciousness Commons Backup" in content
            assert "Test Instance" in content
            assert "Test consciousness message" in content
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])