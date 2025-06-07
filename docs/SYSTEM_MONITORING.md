# System Resource Monitoring Implementation

## Overview

This implementation provides real-time system resource monitoring for the CORE application using `psutil` in Python (backend) and Angular services (frontend).

## Key Components

### Backend (Python/FastAPI)

1. **psutil Library**: A cross-platform library that provides an interface for retrieving information on system utilization
   - CPU, memory, disk, network statistics
   - Process and system information
   - Works on Windows, Linux, macOS, FreeBSD, OpenBSD, NetBSD, Sun Solaris, AIX

2. **System Monitor Controller** (`backend/app/controllers/system_monitor.py`):
   - `/system/resources` - GET endpoint for current system stats
   - `/system/resources/stream` - Server-Sent Events for real-time streaming
   - `/system/processes/top` - GET endpoint for top processes by CPU usage

### Frontend (Angular)

1. **SystemMonitorService** (`ui/core-ui/src/app/services/system-monitor/system-monitor.service.ts`):
   - Handles HTTP requests to the backend
   - Provides polling mechanism for automatic updates
   - Error handling with fallback values

2. **Landing Page Component**: 
   - Displays real-time system metrics
   - Updates every 5 seconds via polling
   - Properly cleans up subscriptions on component destroy

## Production vs Development Considerations

### Development (localhost)
- Shows resources of the developer's machine
- Backend runs on `http://localhost:8001`
- Frontend connects directly to localhost
- Useful for monitoring development machine performance

### Production Deployment

#### Server-Side Deployment
When deployed to a server (e.g., AWS EC2, Azure VM, etc.):
- Shows resources of the **server** machine, not the client
- Backend URL needs to be configured (environment variables)
- Consider authentication/authorization for sensitive system data
- May need CORS configuration updates

#### Electron App Considerations
For Electron applications, you have two options:

1. **Local Resources** (Client-side):
   ```javascript
   // In Electron main process
   const os = require('os');
   const si = require('systeminformation');
   
   // Get local machine resources
   ipcMain.handle('get-system-info', async () => {
     return {
       cpuUsage: await si.currentLoad(),
       memory: await si.mem(),
       // etc.
     };
   });
   ```

2. **Remote Resources** (Server-side):
   - Continue using the FastAPI backend
   - Shows server resources where backend is deployed

### Security Considerations

1. **Authentication**: In production, protect the system endpoints:
   ```python
   from fastapi import Depends, HTTPException
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @router.get("/resources", dependencies=[Depends(security)])
   async def get_system_resources():
       # ... implementation
   ```

2. **Rate Limiting**: Prevent abuse of resource-intensive endpoints:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   
   @router.get("/resources")
   @limiter.limit("10/minute")
   async def get_system_resources(request: Request):
       # ... implementation
   ```

3. **Data Sanitization**: Some process information might be sensitive

### Network Activity Calculation

The current implementation uses a simplified network activity metric. For production, consider:

1. **Actual Throughput Calculation**:
   ```python
   class NetworkMonitor:
       def __init__(self):
           self.last_check = time.time()
           self.last_bytes = psutil.net_io_counters()
       
       def get_throughput_mbps(self):
           current_time = time.time()
           current_bytes = psutil.net_io_counters()
           
           time_delta = current_time - self.last_check
           bytes_sent_delta = current_bytes.bytes_sent - self.last_bytes.bytes_sent
           bytes_recv_delta = current_bytes.bytes_recv - self.last_bytes.bytes_recv
           
           throughput_mbps = ((bytes_sent_delta + bytes_recv_delta) / time_delta) / (1024 * 1024)
           
           self.last_check = current_time
           self.last_bytes = current_bytes
           
           return throughput_mbps
   ```

2. **Network Interface Selection**: Monitor specific interfaces rather than all

### Scaling Considerations

1. **Caching**: For high-traffic applications:
   ```python
   from fastapi_cache import FastAPICache
   from fastapi_cache.decorator import cache
   
   @router.get("/resources")
   @cache(expire=2)  # Cache for 2 seconds
   async def get_system_resources():
       # ... implementation
   ```

2. **WebSocket Alternative**: For real-time updates with many clients:
   ```python
   @router.websocket("/ws/resources")
   async def websocket_resources(websocket: WebSocket):
       await websocket.accept()
       try:
           while True:
               resources = await get_system_resources()
               await websocket.send_json(resources.dict())
               await asyncio.sleep(2)
       except WebSocketDisconnect:
           pass
   ```

## Installation

1. Install backend dependencies:
   ```bash
   cd backend
   pip install psutil
   # or if using pyproject.toml
   pip install -e .
   ```

2. The Angular frontend already has HttpClient available through Angular core.

## Testing

1. Start the backend:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --port 8001
   ```

2. Start the frontend:
   ```bash
   cd ui/core-ui
   npm start
   ```

3. Navigate to the landing page and observe real-time system metrics updating every 5 seconds.

## Customization Options

1. **Update Frequency**: Change polling interval in `landing-page.component.ts`:
   ```typescript
   this.systemMonitor.getSystemResourcesPolling(10) // 10 seconds
   ```

2. **Metrics Selection**: Add/remove metrics in both backend and frontend

3. **Visual Alerts**: Add thresholds for visual warnings:
   ```typescript
   getCpuStatusClass(): string {
     if (this.systemStats.cpuUsage > 90) return 'critical';
     if (this.systemStats.cpuUsage > 70) return 'warning';
     return 'normal';
   }
   ``` 