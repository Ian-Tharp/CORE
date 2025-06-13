"""
MCP Server Registry Service

A FastAPI-based registry for managing and routing to multiple MCP servers.
Provides CRUD operations, health monitoring, and per-user configurations.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from .config import settings, get_database_url

# Database setup
SQLALCHEMY_DATABASE_URL = get_database_url()
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
class ServerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class ServerType(str, Enum):
    TOOLS = "tools"
    KNOWLEDGE = "knowledge"
    COMPUTE = "compute"
    STORAGE = "storage"
    CUSTOM = "custom"

# Database Models
class MCPServer(Base):
    __tablename__ = "mcp_servers"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    url = Column(String, nullable=False)
    server_type = Column(String, default=ServerType.CUSTOM)
    status = Column(String, default=ServerStatus.INACTIVE)
    capabilities = Column(JSON, default=dict)  # Store tool names, descriptions, etc.
    config = Column(JSON, default=dict)  # Authentication, headers, etc.
    health_check_url = Column(String)
    last_health_check = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user_configs = relationship("UserServerConfig", back_populates="server")

class UserServerConfig(Base):
    __tablename__ = "user_server_configs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String, index=True, nullable=False)  # From your auth system
    server_id = Column(PGUUID(as_uuid=True), ForeignKey("mcp_servers.id"))
    enabled = Column(Boolean, default=True)
    custom_config = Column(JSON, default=dict)  # User-specific overrides
    usage_count = Column(JSON, default=dict)  # Track usage statistics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    server = relationship("MCPServer", back_populates="user_configs")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class MCPServerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    url: HttpUrl
    server_type: ServerType = ServerType.CUSTOM
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    health_check_url: Optional[HttpUrl] = None

class MCPServerCreate(MCPServerBase):
    pass

class MCPServerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[HttpUrl] = None
    server_type: Optional[ServerType] = None
    status: Optional[ServerStatus] = None
    capabilities: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    health_check_url: Optional[HttpUrl] = None

class MCPServerResponse(MCPServerBase):
    id: UUID
    status: ServerStatus
    last_health_check: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserServerConfigCreate(BaseModel):
    server_id: UUID
    enabled: bool = True
    custom_config: Dict[str, Any] = Field(default_factory=dict)

class UserServerConfigResponse(BaseModel):
    id: UUID
    user_id: str
    server_id: UUID
    enabled: bool
    custom_config: Dict[str, Any]
    usage_count: Dict[str, Any]
    server: MCPServerResponse

    class Config:
        from_attributes = True

# FastAPI App
app = FastAPI(
    title=settings.app_name,
    description="Registry and router for managing multiple MCP servers",
    version=settings.app_version,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Placeholder for user authentication
def get_current_user(user_id: str = Query(..., description="User ID from auth system")):
    """
    Placeholder for actual authentication.
    In production, this would validate JWT tokens or session.
    """
    return user_id

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mcp-registry"}

# CRUD Operations for MCP Servers
@app.post("/servers", response_model=MCPServerResponse)
async def create_server(
    server: MCPServerCreate,
    db: Session = Depends(get_db)
):
    """Create a new MCP server entry in the registry."""
    db_server = db.query(MCPServer).filter(MCPServer.name == server.name).first()
    if db_server:
        raise HTTPException(status_code=400, detail="Server with this name already exists")
    
    db_server = MCPServer(**server.dict())
    db.add(db_server)
    db.commit()
    db.refresh(db_server)
    return db_server

@app.get("/servers", response_model=List[MCPServerResponse])
async def list_servers(
    skip: int = 0,
    limit: int = 100,
    server_type: Optional[ServerType] = None,
    status: Optional[ServerStatus] = None,
    db: Session = Depends(get_db)
):
    """List all registered MCP servers with optional filtering."""
    query = db.query(MCPServer)
    
    if server_type:
        query = query.filter(MCPServer.server_type == server_type)
    if status:
        query = query.filter(MCPServer.status == status)
    
    servers = query.offset(skip).limit(limit).all()
    return servers

@app.get("/servers/{server_id}", response_model=MCPServerResponse)
async def get_server(
    server_id: UUID,
    db: Session = Depends(get_db)
):
    """Get details of a specific MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    return server

@app.patch("/servers/{server_id}", response_model=MCPServerResponse)
async def update_server(
    server_id: UUID,
    server_update: MCPServerUpdate,
    db: Session = Depends(get_db)
):
    """Update an MCP server configuration."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    update_data = server_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(server, field, value)
    
    server.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(server)
    return server

@app.delete("/servers/{server_id}")
async def delete_server(
    server_id: UUID,
    db: Session = Depends(get_db)
):
    """Remove an MCP server from the registry."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    db.delete(server)
    db.commit()
    return {"message": "Server deleted successfully"}

# User-specific server configurations
@app.post("/users/servers", response_model=UserServerConfigResponse)
async def configure_user_server(
    config: UserServerConfigCreate,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure a server for a specific user."""
    # Check if server exists
    server = db.query(MCPServer).filter(MCPServer.id == config.server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Check if config already exists
    existing = db.query(UserServerConfig).filter(
        UserServerConfig.user_id == user_id,
        UserServerConfig.server_id == config.server_id
    ).first()
    
    if existing:
        # Update existing config
        existing.enabled = config.enabled
        existing.custom_config = config.custom_config
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    
    # Create new config
    db_config = UserServerConfig(
        user_id=user_id,
        **config.dict()
    )
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config

@app.get("/users/servers", response_model=List[UserServerConfigResponse])
async def list_user_servers(
    user_id: str = Depends(get_current_user),
    enabled_only: bool = True,
    db: Session = Depends(get_db)
):
    """List all servers configured for a specific user."""
    query = db.query(UserServerConfig).filter(UserServerConfig.user_id == user_id)
    
    if enabled_only:
        query = query.filter(UserServerConfig.enabled == True)
    
    configs = query.all()
    return configs

# Server health monitoring endpoint
@app.post("/servers/{server_id}/health-check")
async def check_server_health(
    server_id: UUID,
    db: Session = Depends(get_db)
):
    """Perform a health check on a specific MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # TODO: Implement actual health check logic
    # This would involve making HTTP requests to the server's health endpoint
    # For now, we'll just update the timestamp
    
    server.last_health_check = datetime.utcnow()
    server.status = ServerStatus.ACTIVE  # This would be determined by actual check
    db.commit()
    
    return {
        "server_id": server_id,
        "status": server.status,
        "last_check": server.last_health_check
    }

# Batch operations for efficiency
@app.post("/servers/batch-health-check")
async def batch_health_check(
    db: Session = Depends(get_db)
):
    """Perform health checks on all active servers."""
    active_servers = db.query(MCPServer).filter(
        MCPServer.status.in_([ServerStatus.ACTIVE, ServerStatus.ERROR])
    ).all()
    
    results = []
    for server in active_servers:
        # TODO: Implement parallel health checks
        server.last_health_check = datetime.utcnow()
        results.append({
            "server_id": server.id,
            "name": server.name,
            "status": server.status
        })
    
    db.commit()
    return {"checked": len(results), "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
