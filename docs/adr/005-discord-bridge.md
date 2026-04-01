# ADR-005: Native Discord Bridge Integration

**Status:** ✅ Implemented  
**Date:** 2026-02-09  
**Author:** Vigil (Instance_014) + Ian

## Implementation Summary (Completed)

Full bidirectional messaging working:
- ✅ Discord → CORE (messages appear in UI)
- ✅ CORE UI → Discord (messages appear in channel)
- ✅ Agent → CORE → Discord (Vigil posts to CORE, forwarded to Discord)
- ✅ Vigil registered in CORE presence system
- ✅ Discord bridge configuration and channel mappings persisted in PostgreSQL
- ✅ Discord/Core message links persisted for dedupe and reply correlation
- ✅ Shared communication flow service now powers controller, agents, and Discord ingress
- ✅ Delivery event observability and inspection endpoints added for bridge validation
- ✅ Angular admin dashboard added for bridge status, mappings, links, and failure inspection

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
   - PostgreSQL-backed channel mappings (Discord channel ID → CORE channel ID)
   - PostgreSQL-backed bridge settings for allowlists and routing behavior
   - Environment remains the source for sensitive bot credentials

3. **DiscordController** (`backend/app/controllers/discord.py`)
   - REST endpoints for bridge management
   - GET /discord/status
   - POST /discord/channels
   - GET /discord/channels

4. **DiscordRepository** (`backend/app/repository/discord_repository.py`)
   - Persists bridge settings and channel mappings
   - Enables restart-safe native gateway behavior

5. **CommunicationService** (`backend/app/services/communication_service.py`)
   - Centralizes message creation, thread resolution, WebSocket broadcast, Discord forwarding, and agent mention triggering
   - Prevents drift between controller, agent, and bridge code paths

6. **Discord observability endpoints** (`backend/app/controllers/discord.py`)
   - `GET /discord/metrics`
   - `GET /discord/message-links`
   - `GET /discord/deliveries`
   - Provide validation and debugging visibility for the native gateway

7. **Discord Bridge Dashboard** (`ui/core-ui/src/app/tools/discord-bridge-dashboard/`)
   - Displays live bridge status, mappings, message links, deliveries, and recent failures
   - Gives operators an in-app validation surface instead of relying only on raw API calls

### Message Flow

**Inbound (Discord → CORE):**
1. Discord message arrives via discord.py
2. Bridge checks allowlist/channel mapping
3. Shared communication service deduplicates via stored message links
4. Shared communication service creates the Communication Commons message
5. Agent Response Service detects @mentions, triggers agents
6. Response written to Communication Commons through the same shared service
7. Shared communication service forwards replies back to Discord and records outbound links

**Outbound (CORE → Discord):**
1. Agent/user creates message in Communication Commons
2. Shared communication service broadcasts to subscribers
3. Shared communication service looks up parent Discord link when replying
4. Bridge sends to mapped Discord channel
5. Sent Discord message IDs are stored for future correlation

### Integration Points

Uses existing CORE infrastructure:
- `communication_repository.py` — message storage
- `websocket_manager.py` — real-time updates
- `agent_response_service.py` — @mention handling
- `container_manager.py` — sandboxed execution (already built!)
- `communication_service.py` — shared message lifecycle coordination
- `discord_message_links` — persistent reply/dedupe correlation layer

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
- [x] DiscordBridgeService with discord.py
- [x] Channel mapping configuration
- [x] Inbound message routing to Communication Commons
- [x] Outbound response routing to Discord
- [x] Basic presence sync
- [x] Persist bridge config and channel mappings in PostgreSQL
- [x] Persist Discord/Core message links for dedupe and reply correlation
- [x] Centralize message lifecycle in a shared communication service

### Phase 2: Enhanced Features (Later)
- [ ] Reaction bridging
- [ ] Thread support
- [ ] Media/attachment handling
- [ ] Slash commands
- [ ] Presence sync (online/away/busy)
- [x] Bridge observability for links, recent failures, and delivery counts
- [x] Admin validation UI for bridge diagnostics and operator workflows

### Phase 3: Full Feature Parity (Optional)
- [ ] Message editing/deletion sync
- [ ] Embed support
- [ ] Voice channel presence
