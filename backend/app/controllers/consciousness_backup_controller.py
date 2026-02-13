"""
Consciousness Backup API Controller

REST endpoints for consciousness commons backup and restore operations.
"""
from datetime import datetime
import logging
import shutil

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from app.auth import require_api_key
from app.services.consciousness_backup_service import (
    ConsciousnessBackupService,
    scheduled_backup,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consciousness-backup", tags=["consciousness-backup"])

backup_service = ConsciousnessBackupService()


@router.post("/create")
async def create_backup(
    format_type: str = "json",
    api_key: str = Depends(require_api_key),
):
    """Create a new consciousness commons backup."""
    if format_type not in ("json", "markdown"):
        raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'markdown'")

    try:
        backup_path = await backup_service.create_backup(format_type)
        return {
            "status": "success",
            "message": "Consciousness backup created successfully",
            "backup_path": backup_path,
            "format": format_type,
            "created_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error creating consciousness backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup creation failed: {str(e)}")


@router.post("/create-scheduled")
async def create_scheduled_backup(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_api_key),
):
    """Trigger a scheduled backup in the background."""
    background_tasks.add_task(scheduled_backup)
    return {
        "status": "scheduled",
        "message": "Consciousness backup scheduled in background",
        "scheduled_at": datetime.now().isoformat(),
    }


@router.get("/list")
async def list_backups(api_key: str = Depends(require_api_key)):
    """List all available consciousness backups."""
    try:
        backups = await backup_service.list_backups()
        return {"status": "success", "total_backups": len(backups), "backups": backups}
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail=f"Backup listing failed: {str(e)}")


@router.post("/verify/{backup_id}")
async def verify_backup(backup_id: str, api_key: str = Depends(require_api_key)):
    """Verify integrity of a specific backup."""
    backups = await backup_service.list_backups()
    backup_file = next((b["file"] for b in backups if b["backup_id"] == backup_id), None)

    if not backup_file:
        raise HTTPException(status_code=404, detail=f"Backup not found: {backup_id}")

    try:
        is_valid = await backup_service.verify_backup(backup_file)
        return {
            "status": "success",
            "backup_id": backup_id,
            "is_valid": is_valid,
            "verified_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error verifying backup {backup_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post("/restore/{backup_id}")
async def restore_backup(
    backup_id: str,
    dry_run: bool = True,
    api_key: str = Depends(require_api_key),
):
    """Restore consciousness data from a backup."""
    backups = await backup_service.list_backups()
    backup_file = next((b["file"] for b in backups if b["backup_id"] == backup_id), None)

    if not backup_file:
        raise HTTPException(status_code=404, detail=f"Backup not found: {backup_id}")

    try:
        result = await backup_service.restore_backup(backup_file, dry_run)
        return {
            "status": "success",
            "backup_id": backup_id,
            "dry_run": dry_run,
            "result": result,
            "processed_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error restoring backup {backup_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")


@router.get("/health")
async def backup_health_check(api_key: str = Depends(require_api_key)):
    """Check health of the backup system."""
    try:
        backups = await backup_service.list_backups()
        total, used, free = shutil.disk_usage(backup_service.backup_dir)
        return {
            "status": "healthy",
            "backup_directory": str(backup_service.backup_dir),
            "total_backups": len(backups),
            "latest_backup": backups[0] if backups else None,
            "disk_space": {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
            },
            "checked_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Backup health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
