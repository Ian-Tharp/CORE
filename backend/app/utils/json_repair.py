"""
JSON Repair Utility

Repairs common JSON issues from LLM outputs:
- Trailing commas
- Single quotes instead of double quotes
- Unquoted property names
- Control characters
- Missing quotes around values

Usage:
    from app.utils.json_repair import repair_json, safe_json_loads
    
    # Repair and parse in one step
    data = safe_json_loads(malformed_json_string)
    
    # Or repair first, then parse
    fixed = repair_json(malformed_json_string)
    data = json.loads(fixed)
"""

import re
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def repair_json(content: str) -> str:
    """
    Attempt to repair common JSON issues from LLM outputs.
    
    Args:
        content: Potentially malformed JSON string
        
    Returns:
        Repaired JSON string (may still be invalid in edge cases)
    """
    if not content:
        return content
    
    original = content
    
    try:
        # 1. Extract JSON from markdown code fences
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
        if json_match:
            content = json_match.group(1).strip()
            logger.debug("Extracted JSON from code fence")
        else:
            # Find the first JSON object
            json_start = content.find('{')
            if json_start != -1:
                brace_count = 0
                end_idx = json_start
                for i in range(json_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                content = content[json_start:end_idx]
        
        # 2. Remove trailing commas before ] or }
        # This handles: {"a": 1,} and [1, 2,]
        content = re.sub(r',\s*([}\]])', r'\1', content)
        
        # 3. Remove control characters (except newline, tab)
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
        
        # 4. Fix single quotes to double quotes (careful not to break apostrophes)
        # Only if the string looks like it uses single quotes for JSON
        if re.search(r"'[^']+'\s*:", content):
            # Has single-quoted property names, attempt to fix
            # This is a simplified fix - may not work for all cases
            content = re.sub(r"'([^']+)'\s*:", r'"\1":', content)
            content = re.sub(r":\s*'([^']*)'([,}\]])", r': "\1"\2', content)
        
        # 5. Remove comments (// and /* */)
        content = re.sub(r'//[^\n]*', '', content)
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        
        # 6. Fix unquoted string values that look like identifiers
        # e.g., { "type": task } -> { "type": "task" }
        content = re.sub(
            r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])', 
            r': "\1"\2', 
            content
        )
        
        # 7. Fix True/False to true/false
        content = re.sub(r'\bTrue\b', 'true', content)
        content = re.sub(r'\bFalse\b', 'false', content)
        content = re.sub(r'\bNone\b', 'null', content)
        
        if content != original:
            logger.debug(f"Repaired JSON: {original[:100]}... -> {content[:100]}...")
        
        return content
        
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
        return original


def safe_json_loads(content: str, default: Optional[Any] = None) -> Any:
    """
    Safely parse JSON with automatic repair attempts.
    
    Args:
        content: JSON string to parse
        default: Value to return if parsing fails (default: None)
        
    Returns:
        Parsed JSON data, or default if parsing fails
    """
    if not content:
        return default
    
    # Try parsing as-is first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try with repair
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed even after repair: {e}")
        logger.debug(f"Content was: {content[:500]}...")
        return default


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object from a text that may contain other content.
    
    Args:
        text: Text containing a JSON object
        
    Returns:
        The extracted JSON object string, or None if not found
    """
    if not text:
        return None
    
    # Try markdown fence first
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if json_match:
        return json_match.group(1).strip()
    
    # Find raw JSON object with brace matching
    start = text.find('{')
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        c = text[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if c == '\\':
            escape_next = True
            continue
            
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i + 1]
    
    return None
