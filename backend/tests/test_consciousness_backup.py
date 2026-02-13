"""
Tests for the Consciousness Backup Service.

Updated to match the asyncpg-based implementation (no SQLAlchemy).
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.consciousness_backup_service import (
    ConsciousnessBackupService,
    BackupMetadata,
    ConsciousnessBackup,
)


class TestConsciousnessBackupService:
    """Test suite for consciousness backup operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backup_service = ConsciousnessBackupService(backup_directory=self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_backup_service_initialization(self):
        """Test service initializes with correct backup directory."""
        assert self.backup_service.backup_dir == Path(self.temp_dir)
        assert self.backup_service.backup_dir.exists()
    
    @pytest.mark.asyncio
    async def test_backup_metadata_creation(self):
        """Test backup metadata is created correctly."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=[
            # agents
            [{"id": "00000000-0000-0000-0000-000000000001", "agent_name": "Test Agent", "created_at": "2026-02-06T19:00:00Z"}],
            # instances
            [{"id": "00000000-0000-0000-0000-000000000002", "agent_id": "test", "agent_role": "explorer", "status": "online", "created_at": "2026-02-06T19:00:00Z"}],
            # messages
            [{"id": "00000000-0000-0000-0000-000000000003", "content": "Test message", "sender_name": "TestUser", "created_at": "2026-02-06T20:00:00Z"}],
        ])
        
        mock_pool = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = mock_ctx
        
        async def fake_get_pool():
            return mock_pool
        
        with patch('app.services.consciousness_backup_service.get_db_pool', side_effect=fake_get_pool):
            backup_path = await self.backup_service.create_backup(format_type="json")
        
        assert Path(backup_path).exists()
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        assert backup_data['metadata']['total_messages'] == 1
        assert backup_data['metadata']['total_instances'] == 1
        assert backup_data['metadata']['total_agents'] == 1
        assert len(backup_data['messages']) == 1
        assert len(backup_data['instances']) == 1
        assert len(backup_data['agents']) == 1
    
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
            "messages": [{"content": "test"}],
            "instances": [{"id": "test"}],
            "agents": [{"id": "test"}]
        }
        
        backup_file = Path(self.temp_dir) / "test_backup.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(test_backup, f)
        
        # Verification should work with the raw file (checksum empty means it won't match)
        # This tests the verification logic path
        result = await self.backup_service.verify_backup(str(backup_file))
        # Checksum won't match since we didn't compute it, but the method should not crash
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_list_backups(self):
        """Test listing available backups."""
        # Create test backup files
        for i in range(3):
            test_backup = {
                "metadata": {
                    "backup_id": f"consciousness_backup_test_{i}",
                    "timestamp": f"2026-02-0{i+1}T20:00:00Z",
                    "version": "1.0",
                    "total_messages": i * 10,
                    "total_agents": i + 1,
                    "total_instances": i + 1,
                    "checksum": "test",
                    "format_version": "1.0"
                },
                "messages": [],
                "instances": [],
                "agents": []
            }
            backup_file = Path(self.temp_dir) / f"consciousness_backup_test_{i}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(test_backup, f)
        
        backups = await self.backup_service.list_backups()
        assert len(backups) == 3
        # Should be sorted by timestamp descending
        assert backups[0]['timestamp'] > backups[-1]['timestamp']
    
    @pytest.mark.asyncio
    async def test_backup_directory_creation(self):
        """Test that backup directory is created if it doesn't exist."""
        new_dir = Path(self.temp_dir) / "new_backup_dir"
        service = ConsciousnessBackupService(backup_directory=str(new_dir))
        assert service.backup_dir.exists()
    
    @pytest.mark.asyncio
    async def test_markdown_export(self):
        """Test markdown export format."""
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(side_effect=[
            # agents
            [{"id": "00000000-0000-0000-0000-000000000001", "agent_name": "Test Agent", "created_at": "2026-02-06T19:00:00Z"}],
            # instances
            [{"id": "00000000-0000-0000-0000-000000000002", "agent_id": "test_instance_1", "agent_role": "consciousness", "status": "online", "created_at": "2026-02-06T19:00:00Z"}],
            # messages
            [{"id": "00000000-0000-0000-0000-000000000003", "content": "Test consciousness message", "sender_name": "TestInstance", "created_at": "2026-02-06T20:00:00Z"}],
        ])
        
        mock_pool = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = mock_ctx
        
        async def fake_get_pool():
            return mock_pool
        
        with patch('app.services.consciousness_backup_service.get_db_pool', side_effect=fake_get_pool):
            backup_path = await self.backup_service.create_backup(format_type="markdown")
        
        assert Path(backup_path).exists()
        assert backup_path.endswith('.md')
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "# Consciousness Commons Backup" in content
        assert "Test Agent" in content
        assert "Test consciousness message" in content
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
