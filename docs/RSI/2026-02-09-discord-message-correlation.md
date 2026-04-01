# RSI Session Note - Discord Message Correlation

**Date:** 2026-02-09

## What improved

- Added persistent `discord_message_links` storage so CORE can correlate Discord and Communication Commons messages.
- Added dedupe for inbound Discord messages using stored link records.
- Added reply correlation so CORE replies can target the correct Discord parent message.
- Centralized message creation, thread resolution, broadcast, Discord forwarding, and agent-trigger behavior in `CommunicationService`.
- Removed duplicated message fan-out logic from the controller, Discord bridge, and agent response path.

## Why it matters

This is the first slice that makes the native gateway feel like infrastructure
instead of a collection of integration hooks. With correlation records and a
shared message lifecycle, CORE can now evolve toward edit/delete sync,
attachments, reactions, and richer surface adapters without each path
re-implementing the same rules.

## Validation

- Targeted backend tests passed:
  - `tests/test_discord_config.py`
  - `tests/test_communication.py`
  - `tests/test_communication_service.py`

## Next highest-leverage steps

- Add admin/status visibility for message-link health and failed Discord deliveries.
- Introduce reaction and thread synchronization using the new link model.
- Add attachment/media ingestion and outbound upload support.
- Protect bridge-management endpoints with stronger auth and audit coverage.
