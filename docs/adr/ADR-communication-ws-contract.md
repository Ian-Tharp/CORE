### ADR: Normalize Communication WebSocket Event Contract

Status: Accepted
Date: 2025-11-06

Context
- Backend emitted different event types for new messages:
  - REST-sent messages: `type: "message"` with `{ channel_id, message }`
  - Agent responses: `type: "new_message"` without `channel_id` at the top level
- UI only listened to `type: "message"`, creating inconsistency and missed updates for agent responses.

Decision
- Canonicalize all “new message” broadcasts to `type: "message"` and payload:
  - `{ "type": "message", "channel_id": "<channel_id>", "message": { ... } }`
- During migration, dual-emit legacy events behind a feature flag to avoid breaking older clients:
  - `DUAL_WS_MESSAGE_EVENT = True` in `backend/app/services/agent_response_service.py`.
- Emit reaction removal updates via `type: "reaction_removed"` to complement existing `reaction_added`.
- UI listens to both `message` (canonical) and `new_message` (legacy) until flag removal.

Changes
- Backend
  - `agent_response_service.py`: emit canonical `message` and legacy `new_message` when flag is true.
  - `communication.py`: broadcast `reaction_removed` on DELETE reaction.
- UI
  - `WebSocketService`: builds WS URL from `AppConfigService.apiBaseUrl` (http→ws).
  - `CommunicationComponent`: listens to both `message` and `new_message`; handles `reaction_removed`.

Consequences
- Immediate consistency for UI updates regardless of message source.
- Safe rollout via dual-emit; simple removal path later.

Deprecation Plan
- Keep `DUAL_WS_MESSAGE_EVENT=True` for one release.
- Remove legacy `new_message` emission and UI listener in a subsequent cleanup after release validation.

Verification
- Send message via REST → UI receives `message` event.
- Mention agent → agent response arrives as `message` (and `new_message` during migration).
- Add/remove reactions → UI updates via `reaction_added`/`reaction_removed`.

