# ADR-005: Native Discord Bridge Integration

**Status:** Accepted  
**Date:** 2026-02-09  
**Author:** Vigil (Instance_014) + Ian

## Context

CORE is designed to be a self-hosted JARVIS — a cognitive orchestration platform. Currently, interaction happens through:
1. The Angular/Electron UI (when at desktop)
2. OpenClaw gateway → Discord (when mobile)

The dependency on OpenClaw for Discord creates limitations:
- Feature updates require waiting on npm package releases
- Cannot self-improve the gateway layer
- Two systems to maintain
- Added latency from the hop

## Decision

Build a **native Discord bridge** directly in CORE that connects Discord channels to Communication Commons channels bidirectionally.

```
Discord Server
      │
      ▼
CORE Discord Bridge Service (discord.py)
      │
      ├── Discord events → Communication Commons messages
      │
      ▼
Communication Commons (channels, messages, reactions, presence)
      │
      ▼
CORE Engine / Agent Factory / Council / Bus
      │
      ▼ (response flows back through bridge)
Discord Server
```

## Architecture

### New Components

1. **DiscordBridgeService** (`backend/app/services/discord_bridge.py`)
   - Manages discord.py bot connection
   - Handles Discord events (on_message, on_reaction_add, etc.)
   - Routes messages to/from Communication Commons

2. **DiscordConfig** (`backend/app/config/discord.py`)
   - Bot token (from environment)
   - Channel mappings (Discord channel ID → CORE channel ID)
   - User allowlists
   - Feature flags

3. **DiscordController** (`backend/app/controllers/discord.py`)
   - REST endpoints for bridge management
   - GET /discord/status
   - POST /discord/channels/map
   - GET /discord/channels

### Message Flow

**Inbound (Discord → CORE):**
1. Discord message arrives via discord.py
2. Bridge checks allowlist/channel mapping
3. Bridge creates Communication Commons message via existing API
4. Agent Response Service detects @mentions, triggers agents
5. Response written to Communication Commons
6. Bridge's WebSocket subscription catches response
7. Bridge sends to Discord

**Outbound (CORE → Discord):**
1. Agent/user creates message in Communication Commons
2. WebSocket broadcasts to subscribers
3. Bridge receives via WebSocket subscription
4. Bridge sends to mapped Discord channel

### Integration Points

Uses existing CORE infrastructure:
- `communication_repository.py` — message storage
- `websocket_manager.py` — real-time updates
- `agent_response_service.py` — @mention handling
- `container_manager.py` — sandboxed execution (already built!)

## Configuration

```python
# Environment variables
DISCORD_BOT_TOKEN=xxx
DISCORD_ENABLED=true

# Channel mapping (stored in DB or config)
channel_mappings = {
    "1466075379075911821": "discord_updates",  # Discord channel → CORE channel
    "1469753891942961334": "meryems_channel",
}

# User allowlist
allowed_users = ["155385542165397504", "1469753119981179173"]
```

## Consequences

**Positive:**
- Full ownership of the gateway
- Single system to maintain
- Deep integration with Communication Commons
- Self-improvement capability (Vigil can help write features)
- All CORE features available on Discord natively

**Negative:**
- Need to maintain Discord protocol handling
- Initial development effort (~1-2 days)

**Mitigations:**
- discord.py is mature and well-documented
- We can reference OpenClaw's patterns for edge cases
- Start with basic text messaging, add reactions/threads incrementally

## Implementation Plan

### Phase 1: Core Bridge (Today)
- [ ] DiscordBridgeService with discord.py
- [ ] Channel mapping configuration
- [ ] Inbound message routing to Communication Commons
- [ ] Outbound response routing to Discord
- [ ] Basic presence sync

### Phase 2: Enhanced Features (Later)
- [ ] Reaction bridging
- [ ] Thread support
- [ ] Media/attachment handling
- [ ] Slash commands
- [ ] Presence sync (online/away/busy)

### Phase 3: Full Feature Parity (Optional)
- [ ] Message editing/deletion sync
- [ ] Embed support
- [ ] Voice channel presence
