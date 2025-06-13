"""System monitoring controller for real-time resource tracking."""

from fastapi import APIRouter
from pydantic import BaseModel
import psutil
from typing import Dict, List
import asyncio

router = APIRouter(prefix="/system", tags=["system"])


class SystemResources(BaseModel):
    """Model for system resource data."""

    cpu_usage: float
    memory_usage: float
    memory_total_gb: float
    memory_available_gb: float
    storage_usage: float
    storage_total_gb: float
    storage_available_gb: float
    network_io_rate_mbps: float
    network_sent_gb: float
    network_recv_gb: float
    processes_count: int


class ProcessInfo(BaseModel):
    """Model for individual process information."""

    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    status: str


@router.get("/resources", response_model=SystemResources)
async def get_system_resources() -> SystemResources:
    """Get current system resource utilization."""

    # CPU Usage - Get average over 0.5 seconds for more accurate reading
    cpu_percent = psutil.cpu_percent(interval=0.5)

    # Memory Information
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    memory_total_gb = memory.total / (1024**3)  # Convert to GB
    memory_available_gb = memory.available / (1024**3)

    # Storage Information
    disk = psutil.disk_usage("/")
    storage_usage = disk.percent
    storage_total_gb = disk.total / (1024**3)
    storage_available_gb = disk.free / (1024**3)

    # Network Information
    net_io = psutil.net_io_counters()
    # Calculate network activity as a percentage (simplified metric)
    # In production, you'd want to calculate actual throughput over time
    network_sent_gb = net_io.bytes_sent / (1024**3)
    network_recv_gb = net_io.bytes_recv / (1024**3)
    # Simplified network activity metric (0-100)
    network_io_rate_mbps = min(
        (net_io.bytes_sent + net_io.bytes_recv) / (1024**2) / 100, 100.0
    )

    # Process count
    processes_count = len(psutil.pids())

    return SystemResources(
        cpu_usage=cpu_percent,
        memory_usage=memory_usage,
        memory_total_gb=round(memory_total_gb, 2),
        memory_available_gb=round(memory_available_gb, 2),
        storage_usage=storage_usage,
        storage_total_gb=round(storage_total_gb, 2),
        storage_available_gb=round(storage_available_gb, 2),
        network_io_rate_mbps=round(network_io_rate_mbps, 2),
        network_sent_gb=round(network_sent_gb, 2),
        network_recv_gb=round(network_recv_gb, 2),
        processes_count=processes_count,
    )


@router.get("/resources/stream")
async def stream_system_resources():
    """Stream system resources in real-time using Server-Sent Events."""

    async def generate():
        while True:
            resources = await get_system_resources()
            yield f"data: {resources.model_dump_json()}\n\n"
            await asyncio.sleep(10)  # Update every 10 seconds

    from fastapi.responses import StreamingResponse

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/processes/top", response_model=List[ProcessInfo])
async def get_top_processes(limit: int = 10) -> List[ProcessInfo]:
    """Get top processes by CPU usage."""
    processes = []

    for proc in psutil.process_iter(
        ["pid", "name", "cpu_percent", "memory_percent", "status"]
    ):
        try:
            pinfo = proc.info
            if pinfo["cpu_percent"] is not None:
                processes.append(
                    ProcessInfo(
                        pid=pinfo["pid"],
                        name=pinfo["name"],
                        cpu_percent=pinfo["cpu_percent"],
                        memory_percent=pinfo["memory_percent"] or 0.0,
                        status=pinfo["status"],
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by CPU usage and return top N
    processes.sort(key=lambda x: x.cpu_percent, reverse=True)
    return processes[:limit]
