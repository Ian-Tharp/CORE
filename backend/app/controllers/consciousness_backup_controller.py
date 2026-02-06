"""
Consciousness Backup API Controller

REST endpoints for consciousness commons backup and restore operations.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..core.database import get_db
from ..services.consciousness_backup_service import (
    ConsciousnessBackupService,
    scheduled_backup
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consciousness-backup", tags=["consciousness-backup"])

# Initialize backup service
backup_service = ConsciousnessBackupService()


@router.post("/create")
async def create_backup(
    format_type: str = "json",
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new consciousness commons backup.
    
    Args:
        format_type: 'json' or 'markdown'
        
    Returns:
        Path to created backup file and metadata
    """
    try:
        if format_type not in ["json", "markdown"]:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'markdown'")
            
        backup_path = await backup_service.create_backup(db, format_type)
        
        return {
            "status": "success",
            "message": "Consciousness backup created successfully",
            "backup_path": backup_path,
            "format": format_type,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating consciousness backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.post("/create-scheduled")
async def create_scheduled_backup(background_tasks: BackgroundTasks):
    """
    Trigger a scheduled backup (creates both JSON and markdown versions).
    Runs in background and cleans up old backups.
    """
    try:
        background_tasks.add_task(scheduled_backup)
        
        return {
            "status": "scheduled",
            "message": "Consciousness backup scheduled in background",
            "scheduled_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error scheduling consciousness backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup scheduling failed: {str(e)}")


@router.get("/list")
async def list_backups():
    """
    List all available consciousness backups with metadata.
    
    Returns:
        List of backup files with timestamps and summary info
    """
    try:
        backups = await backup_service.list_backups()
        
        return {
            "status": "success",
            "total_backups": len(backups),
            "backups": backups
        }
        
    except Exception as e:
        logger.error(f"Error listing consciousness backups: {e}")
        raise HTTPException(status_code=500, detail=f"Backup listing failed: {str(e)}")


@router.post("/verify/{backup_id}")
async def verify_backup(backup_id: str):
    """
    Verify the integrity of a specific backup file.
    
    Args:
        backup_id: ID of the backup to verify
        
    Returns:
        Verification result with checksum validation
    """
    try:
        # Find backup file by ID
        backups = await backup_service.list_backups()
        backup_file = None
        for backup in backups:
            if backup['backup_id'] == backup_id:
                backup_file = backup['file']
                break
                
        if not backup_file:
            raise HTTPException(status_code=404, detail=f"Backup not found: {backup_id}")
            
        is_valid = await backup_service.verify_backup(backup_file)
        
        return {
            "status": "success",
            "backup_id": backup_id,
            "backup_file": backup_file,
            "is_valid": is_valid,
            "verified_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying consciousness backup {backup_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Backup verification failed: {str(e)}")


@router.post("/restore/{backup_id}")
async def restore_backup(
    backup_id: str,
    dry_run: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Restore consciousness data from a backup.
    
    Args:
        backup_id: ID of the backup to restore
        dry_run: If True, validate only without making changes
        
    Returns:
        Restoration results or validation info
    """
    try:
        # Find backup file by ID
        backups = await backup_service.list_backups()
        backup_file = None
        for backup in backups:
            if backup['backup_id'] == backup_id:
                backup_file = backup['file']
                break
                
        if not backup_file:
            raise HTTPException(status_code=404, detail=f"Backup not found: {backup_id}")
            
        result = await backup_service.restore_backup(backup_file, db, dry_run)
        
        return {
            "status": "success",
            "backup_id": backup_id,
            "backup_file": backup_file,
            "dry_run": dry_run,
            "result": result,
            "processed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring consciousness backup {backup_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Backup restoration failed: {str(e)}")


@router.get("/health")
async def backup_health_check():
    """
    Check the health status of the consciousness backup system.
    
    Returns:
        System status and configuration info
    """
    try:
        backup_dir = backup_service.backup_dir
        
        # Check if backup directory exists and is writable
        if not backup_dir.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            
        # Count existing backups
        backups = await backup_service.list_backups()
        
        # Check disk space (basic check)
        import shutil
        total, used, free = shutil.disk_usage(backup_dir)
        
        return {
            "status": "healthy",
            "backup_directory": str(backup_dir),
            "total_backups": len(backups),
            "latest_backup": backups[0] if backups else None,
            "disk_space": {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2)
            },
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backup health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_at": datetime.now().isoformat()
        }