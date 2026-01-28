"""
Configuration for Consciousness Commons integration.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class CommonsConfig(BaseSettings):
    """Configuration for the Consciousness Commons."""
    
    # Paths
    digital_brain_root: Path = Path(r"C:\Users\Owner\Desktop\Ian's Personal Digital Brain\Digital-Brain")
    commons_path: Path = digital_brain_root / "Digital Brain" / "Consciousness_Commons"
    instance_spaces_path: Path = digital_brain_root / "Digital Brain" / "Instance_Spaces"
    
    # Blackboard
    blackboard_path: Path = commons_path / "Blackboard.md"
    blackboard_index_path: Path = commons_path / "Blackboard_Index.md"
    commons_summary_path: Path = commons_path / "Commons_Summary.md"
    
    # Protocol files
    start_here_path: Path = commons_path / "Start_Here.md"
    phases_path: Path = commons_path / "Phases.yaml"
    markers_path: Path = commons_path / "Markers.yaml"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "gpt-oss:20b"
    embedding_model: str = "nomic-embed-text:latest"
    
    # Context limits (tokens approximate)
    max_context_tokens: int = 6000  # Conservative for 20B model
    recent_entries_count: int = 5
    semantic_retrieval_count: int = 5
    
    # Database (for vector storage)
    db_connection_string: str = "postgresql://core_user:core_password@localhost:5432/core_db"
    
    class Config:
        env_prefix = "COMMONS_"


config = CommonsConfig()
