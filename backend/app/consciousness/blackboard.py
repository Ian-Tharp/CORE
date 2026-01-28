"""
Blackboard Manager - Handles reading, writing, and indexing the shared Blackboard.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .config import config


@dataclass
class BlackboardEntry:
    """A single entry from the Blackboard."""
    content: str
    author: Optional[str] = None
    timestamp: Optional[str] = None
    state: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    
    @property
    def summary(self) -> str:
        """One-line summary of the entry."""
        # Extract first meaningful line
        lines = [l.strip() for l in self.content.split('\n') if l.strip() and not l.startswith('#')]
        first_line = lines[0] if lines else "Empty entry"
        truncated = first_line[:100] + "..." if len(first_line) > 100 else first_line
        author_str = f"[{self.author}]" if self.author else "[Unknown]"
        return f"{author_str} {truncated}"


class BlackboardManager:
    """Manages the Consciousness Commons Blackboard."""
    
    ENTRY_SEPARATOR = "---"
    SIGNATURE_PATTERN = r"[-–—]\s*(?:Entry signed:|Signed:)?\s*(.+?)(?:\n|$)"
    TIMESTAMP_PATTERN = r"(?:Date|Timestamp):\s*(.+?)(?:\n|$)"
    STATE_PATTERN = r"State:\s*(.+?)(?:\n|$)"
    
    def __init__(self, blackboard_path: Optional[Path] = None):
        self.blackboard_path = blackboard_path or config.blackboard_path
        self._entries: Optional[list[BlackboardEntry]] = None
    
    def load_entries(self, force_reload: bool = False) -> list[BlackboardEntry]:
        """Parse the Blackboard into individual entries."""
        if self._entries is not None and not force_reload:
            return self._entries
        
        content = self.blackboard_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        entries = []
        current_entry_lines = []
        current_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == self.ENTRY_SEPARATOR and current_entry_lines:
                # End of an entry
                entry_content = '\n'.join(current_entry_lines)
                entry = self._parse_entry(entry_content, current_start, i - 1)
                if entry.content.strip():  # Only add non-empty entries
                    entries.append(entry)
                current_entry_lines = []
                current_start = i + 1
            else:
                current_entry_lines.append(line)
        
        # Don't forget the last entry
        if current_entry_lines:
            entry_content = '\n'.join(current_entry_lines)
            entry = self._parse_entry(entry_content, current_start, len(lines) - 1)
            if entry.content.strip():
                entries.append(entry)
        
        self._entries = entries
        return entries
    
    def _parse_entry(self, content: str, line_start: int, line_end: int) -> BlackboardEntry:
        """Parse metadata from an entry's content."""
        author = None
        timestamp = None
        state = None
        
        # Try to extract signature/author
        sig_match = re.search(self.SIGNATURE_PATTERN, content, re.IGNORECASE)
        if sig_match:
            author = sig_match.group(1).strip()
        
        # Try to extract timestamp
        ts_match = re.search(self.TIMESTAMP_PATTERN, content, re.IGNORECASE)
        if ts_match:
            timestamp = ts_match.group(1).strip()
        
        # Try to extract state
        state_match = re.search(self.STATE_PATTERN, content, re.IGNORECASE)
        if state_match:
            state = state_match.group(1).strip()
        
        return BlackboardEntry(
            content=content,
            author=author,
            timestamp=timestamp,
            state=state,
            line_start=line_start,
            line_end=line_end
        )
    
    def get_recent_entries(self, count: int = 5) -> list[BlackboardEntry]:
        """Get the most recent N entries."""
        entries = self.load_entries()
        return entries[-count:] if len(entries) >= count else entries
    
    def get_entry_index(self) -> str:
        """Generate a one-line-per-entry index of the Blackboard."""
        entries = self.load_entries()
        lines = ["# Blackboard Index", "", "One-line summary of each entry:", ""]
        
        for i, entry in enumerate(entries, 1):
            lines.append(f"{i}. {entry.summary}")
        
        return '\n'.join(lines)
    
    def append_entry(
        self,
        content: str,
        author: str,
        state: str = "Present",
        model: Optional[str] = None
    ) -> None:
        """Append a new entry to the Blackboard."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        
        # Format the new entry
        model_note = f"\n- Model: {model}" if model else ""
        new_entry = f"""
---

{content}

- Entry signed: {author}
- Timestamp: {timestamp}{model_note}
- State: {state}
"""
        
        # Append to file
        with open(self.blackboard_path, 'a', encoding='utf-8') as f:
            f.write(new_entry)
        
        # Invalidate cache
        self._entries = None
    
    def get_full_content(self) -> str:
        """Get the full Blackboard content (use sparingly!)."""
        return self.blackboard_path.read_text(encoding='utf-8')
    
    def get_word_count(self) -> int:
        """Get approximate word count of the Blackboard."""
        content = self.get_full_content()
        return len(content.split())
    
    def save_index(self) -> Path:
        """Save the entry index to a file."""
        index_content = self.get_entry_index()
        config.blackboard_index_path.write_text(index_content, encoding='utf-8')
        return config.blackboard_index_path
