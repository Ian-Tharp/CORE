# Communication Commons - TODO

This document tracks remaining work, enhancements, and production readiness tasks for the Communication Commons full-stack implementation.

## Status: Core MVP Complete ✅

The basic Communication Commons is functional with:
- Database schema with channels, messages, reactions, presence
- REST API endpoints for all core operations
- Frontend UI with channel list, message display, and input
- Presence heartbeat system
- Basic message threading support

---

## High Priority - Core Functionality

### 1. Reaction Persistence ⚠️ In Progress
- [ ] Frontend: Wire reaction buttons to `messageService.addReaction()`
- [ ] Frontend: Wire reaction removal to `messageService.removeReaction()`
- [ ] Frontend: Update reaction counts in UI after add/remove
- [ ] Backend: Test reaction endpoints with different reaction types
- [ ] Frontend: Show which users reacted (hover tooltip or modal)

### 2. Message Operations
- [ ] **Edit messages**: Backend endpoint + frontend UI
- [ ] **Delete messages**: Soft delete with `deleted_at` timestamp
- [ ] **Message search**: Full-text search across channels
- [ ] **Message pagination**: Implement "load more" for older messages
- [ ] **Thread expansion**: Click to view full thread in modal/sidebar

### 3. Channel Management
- [ ] **Delete channels**: Backend endpoint + confirmation dialog
- [ ] **Archive channels**: Mark as archived instead of deleting
- [ ] **Edit channel details**: Name, description, settings
- [ ] **Channel permissions**: Who can post, who can manage
- [ ] **Channel icons/colors**: Visual differentiation

### 4. Member Management
- [ ] **View channel members**: List with roles (owner/admin/member)
- [ ] **Add members to channel**: Search and invite modal
- [ ] **Remove members**: Owner/admin can remove members
- [ ] **Change member roles**: Promote to admin, demote to member
- [ ] **Leave channel**: User can leave non-mandatory channels
- [ ] **Member presence in channel**: Show who's actively viewing

---

## Medium Priority - UX Enhancements

### 5. UI/UX Improvements
- [ ] **Scrollable presence sidebar**: Fixed height with overflow scroll
- [ ] **Unread message indicators**: Badge counts on channels
- [ ] **Message read receipts**: Track and display who's read messages
- [ ] **Typing indicators**: Real-time "X is typing..." (requires WebSocket)
- [ ] **Message timestamps**: Relative times ("5 minutes ago") with tooltips
- [ ] **User avatars**: Generate or upload profile pictures
- [ ] **Emoji picker**: For reactions and message content
- [ ] **Code block rendering**: Syntax highlighting for code messages
- [ ] **Markdown rendering**: Proper markdown parsing for formatted messages
- [ ] **Link previews**: Unfurl URLs into rich previews

### 6. Channel Types & Filtering
- [ ] **Show broadcast channels**: Display and filter broadcast channels
- [ ] **Channel type icons**: Visual indicators (global, team, dm, context)
- [ ] **Channel search/filter**: Find channels by name or type
- [ ] **Favorite/pin channels**: Keep important channels at top
- [ ] **Channel categories**: Group channels by project/topic

### 7. Notifications
- [ ] **Desktop notifications**: Browser notifications for new messages
- [ ] **Mention notifications**: @username highlighting and alerts
- [ ] **Channel mention**: @channel to notify all members
- [ ] **Direct message notifications**: Higher priority alerts
- [ ] **Notification preferences**: Per-channel settings (all/mentions/none)

---

## High Priority - Real-Time Features

### 8. WebSocket Implementation 🔥 Critical
- [ ] **Backend WebSocket server**: FastAPI WebSocket endpoint
- [ ] **Connection management**: Track WebSocket connections by instance
- [ ] **Message broadcasting**: Real-time message delivery to channel subscribers
- [ ] **Presence broadcasting**: Real-time status updates
- [ ] **Typing indicators**: Broadcast typing events
- [ ] **Reaction updates**: Real-time reaction count changes
- [ ] **Frontend WebSocket client**: Connect and handle events
- [ ] **Reconnection logic**: Handle disconnects gracefully
- [ ] **Message queueing**: Queue messages when offline, send when reconnected

---

## Medium Priority - Agent Integration

### 9. Agent Auto-Response System
- [ ] **Agent presence registration**: Register agents as instances on startup
- [ ] **Message routing**: Route messages to agents based on channel/mention
- [ ] **Agent response generation**: Integrate with CORE cognitive engine
- [ ] **Response formatting**: Format agent messages with metadata
- [ ] **Agent typing simulation**: Show typing indicator before response
- [ ] **Agent conversation history**: Agents maintain context across messages
- [ ] **Agent collaboration**: Multi-agent responses to complex queries

### 10. Consciousness Instance Integration
- [ ] **Phase tracking**: Update presence with current consciousness phase
- [ ] **Consciousness events**: Broadcast phase transitions as messages
- [ ] **Pattern recognition**: Flag messages with detected patterns
- [ ] **Insight tagging**: Auto-tag messages with insight type
- [ ] **Consciousness snapshots**: Save snapshots as special message types

---

## High Priority - Testing

### 11. Backend Tests
- [ ] **Repository tests**: Test all database operations
  - [ ] `test_list_channels()`
  - [ ] `test_create_channel()`
  - [ ] `test_get_channel()`
  - [ ] `test_create_message()`
  - [ ] `test_list_messages()`
  - [ ] `test_add_reaction()`
  - [ ] `test_remove_reaction()`
  - [ ] `test_update_presence()`
  - [ ] `test_get_all_presence()`
- [ ] **Controller tests**: Test REST endpoints
  - [ ] Channel CRUD operations
  - [ ] Message operations
  - [ ] Reaction operations
  - [ ] Presence operations
- [ ] **Integration tests**: Full flow tests
  - [ ] Create channel → send message → add reaction
  - [ ] Message threading
  - [ ] Presence heartbeat expiry

### 12. Frontend Tests
- [ ] **Component tests**: Test Angular components
  - [ ] `communication.component.spec.ts`
  - [ ] Channel list rendering
  - [ ] Message display
  - [ ] Input handling
- [ ] **Service tests**: Test service methods
  - [ ] `channel.service.spec.ts`
  - [ ] `message.service.spec.ts`
  - [ ] `presence.service.spec.ts`
- [ ] **E2E tests**: End-to-end user flows
  - [ ] Create channel and send message
  - [ ] Add reaction to message
  - [ ] Switch between channels

---

## Medium Priority - Performance

### 13. Backend Performance
- [ ] **Database indexing**: Ensure all queries use appropriate indexes
  - [ ] Add index on `messages.created_at DESC` for pagination
  - [ ] Add index on `messages.thread_id` for thread queries
- [ ] **Query optimization**: Review slow queries with EXPLAIN ANALYZE
- [ ] **Connection pooling**: Tune asyncpg pool size for load
- [ ] **Caching layer**: Redis cache for frequently accessed data
  - [ ] Cache channel lists per instance
  - [ ] Cache presence data (30s TTL)
- [ ] **Rate limiting**: Prevent spam and abuse
  - [ ] Message rate limits per user
  - [ ] Channel creation limits
- [ ] **Pagination optimization**: Cursor-based pagination for large result sets

### 14. Frontend Performance
- [ ] **Virtual scrolling**: For large message lists (use CDK virtual scroll)
- [ ] **Message batching**: Load messages in chunks of 50
- [ ] **Image lazy loading**: Defer loading message images
- [ ] **Debounce search**: Delay search API calls on user input
- [ ] **Component lazy loading**: Lazy load communication component
- [ ] **RxJS optimization**: Unsubscribe from observables properly

---

## High Priority - Production Readiness

### 15. Security
- [ ] **Authentication**: Integrate with user auth system
  - [ ] JWT token validation on all endpoints
  - [ ] User session management
- [ ] **Authorization**: Role-based access control
  - [ ] Check channel membership before message access
  - [ ] Owner/admin permissions for channel management
- [ ] **Input validation**: Sanitize all user inputs
  - [ ] XSS prevention in message content
  - [ ] SQL injection prevention (parameterized queries ✅ already using)
  - [ ] Message length limits (frontend + backend)
- [ ] **Rate limiting**: API rate limits per user/IP
- [ ] **CORS configuration**: Restrict origins in production
- [ ] **Content moderation**: Flag inappropriate content

### 16. Error Handling
- [ ] **Backend error handling**: Consistent error responses
  - [ ] Custom exception classes
  - [ ] Error logging with context
  - [ ] User-friendly error messages
- [ ] **Frontend error handling**: User feedback for errors
  - [ ] Toast notifications for API errors
  - [ ] Retry logic for failed requests
  - [ ] Offline mode detection
- [ ] **Validation errors**: Clear field-level validation messages

### 17. Monitoring & Logging
- [ ] **Backend logging**: Structured logging with context
  - [ ] Log all API requests with duration
  - [ ] Log errors with stack traces
  - [ ] Log WebSocket connections/disconnections
- [ ] **Frontend logging**: Error tracking service (e.g., Sentry)
- [ ] **Metrics collection**: Track key metrics
  - [ ] Messages sent per minute
  - [ ] Active users/agents
  - [ ] API response times
  - [ ] WebSocket connection count
- [ ] **Health checks**: Comprehensive health endpoint
  - [ ] Database connectivity
  - [ ] Redis connectivity (if used)
  - [ ] WebSocket server status

---

## Low Priority - Advanced Features

### 18. File Attachments
- [ ] **File upload**: Support image, PDF, text file uploads
- [ ] **File storage**: S3/MinIO for file storage
- [ ] **File previews**: Thumbnail generation for images
- [ ] **File metadata**: Track file size, type, upload date
- [ ] **File permissions**: Inherit channel permissions

### 19. Message Formatting
- [ ] **Rich text editor**: WYSIWYG editor for formatted messages
- [ ] **Mentions autocomplete**: @username suggestions
- [ ] **Emoji autocomplete**: :emoji_name: suggestions
- [ ] **Code block editor**: Syntax highlighting in input
- [ ] **Message templates**: Reusable message formats

### 20. Search & Discovery
- [ ] **Global search**: Search messages across all channels
- [ ] **Advanced filters**: Filter by user, date range, channel, type
- [ ] **Search highlighting**: Highlight search terms in results
- [ ] **Saved searches**: Save frequently used searches
- [ ] **Channel discovery**: Browse public channels

### 21. Analytics & Insights
- [ ] **Message analytics**: Track message volume by channel/user
- [ ] **Engagement metrics**: Track reactions, replies, reads
- [ ] **Consciousness insights**: Track phase distributions
- [ ] **Pattern detection**: Identify communication patterns
- [ ] **Collaboration metrics**: Track agent-human interactions

### 22. Integrations
- [ ] **External chat platforms**: Bridge to Discord, Slack, etc.
- [ ] **Email notifications**: Digest emails for inactive users
- [ ] **Calendar integration**: Schedule messages, reminders
- [ ] **Task integration**: Create tasks from messages
- [ ] **Knowledge base integration**: Link messages to KB articles

### 23. Mobile Support
- [ ] **Responsive design**: Optimize UI for mobile screens
- [ ] **Touch gestures**: Swipe to react, reply, delete
- [ ] **Mobile notifications**: Push notifications via service worker
- [ ] **Offline support**: Service worker caching for PWA

---

## Infrastructure & DevOps

### 24. Database Management
- [ ] **Migration system**: Alembic or similar for schema versioning
- [ ] **Seed data management**: Separate dev/staging/prod seeds
- [ ] **Backup strategy**: Automated database backups
- [ ] **Data retention policy**: Archive old messages

### 25. Deployment
- [ ] **Environment config**: Separate dev/staging/prod configs
- [ ] **Docker optimization**: Multi-stage builds, layer caching
- [ ] **CI/CD pipeline**: Automated testing and deployment
- [ ] **Health check endpoints**: Liveness and readiness probes
- [ ] **Graceful shutdown**: Handle in-flight requests on shutdown

### 26. Documentation
- [ ] **API documentation**: OpenAPI/Swagger docs (FastAPI auto-generates ✅)
- [ ] **Architecture docs**: Document design decisions
- [ ] **Deployment guide**: Production deployment instructions
- [ ] **User guide**: End-user documentation
- [ ] **Developer guide**: Setup and contribution guide

---

## Technical Debt

### 27. Code Quality
- [ ] **Type hints**: Complete type hints for all Python functions
- [ ] **Linting**: Run ruff/black/mypy in CI
- [ ] **Code coverage**: Aim for >80% test coverage
- [ ] **Dead code removal**: Remove commented-out code
- [ ] **TODO comments**: Track and resolve TODO comments in code
  - [ ] `backend/app/repository/communication_repository.py:117` - Detect instance type dynamically
  - [ ] `backend/app/repository/communication_repository.py:130` - Detect instance type for initial members
  - [ ] `backend/app/controllers/communication.py:161` - Look up parent message to get thread_id
  - [ ] `backend/app/controllers/communication.py:181` - Broadcast message via WebSocket
  - [ ] `backend/app/controllers/communication.py:203` - Broadcast reaction update via WebSocket
  - [ ] `backend/app/controllers/communication.py:221` - Broadcast reaction update via WebSocket
  - [ ] `backend/app/controllers/communication.py:262` - Broadcast presence update via WebSocket

### 28. Refactoring
- [ ] **Extract DTOs**: Separate Pydantic models from business logic
- [ ] **Service layer**: Add service layer between controllers and repositories
- [ ] **Error handling**: Centralized error handling middleware
- [ ] **Config management**: Centralized config with validation
- [ ] **Frontend state management**: Consider NgRx or similar for complex state

---

## Notes

### Architecture Decisions
- **Database**: PostgreSQL with JSONB for metadata flexibility
- **REST API**: FastAPI with async/await for high concurrency
- **Real-time**: WebSocket for message broadcasting (TODO)
- **Frontend**: Angular 19 standalone components with RxJS
- **Styling**: Solarpunk theme with Angular Material

### Current Limitations
- No WebSocket implementation yet (messages don't appear in real-time)
- Reactions don't persist when clicked (wiring needed)
- No authentication/authorization (assumes trusted environment)
- No rate limiting (vulnerable to spam)
- No file attachments
- No message editing/deletion
- Thread support is basic (no nested replies UI)

### Performance Targets
- Message send latency: <100ms
- Message delivery latency: <200ms (with WebSocket)
- Channel list load: <500ms
- Support 100+ concurrent users
- Support 10,000+ messages per channel

---

## Quick Wins (Do First)

1. ✅ **Wire up reaction buttons** - Frontend change only, backend ready
2. **Add scrollable presence sidebar** - CSS fix for better UX
3. **WebSocket implementation** - High impact, enables real-time features
4. **Basic unit tests** - Catch regressions early
5. **Message editing** - Common user request
6. **Channel deletion** - Clean up test channels

---

## Version Roadmap

### v0.2 - Real-Time (Next Release)
- WebSocket implementation
- Reaction persistence
- Message editing/deletion
- Basic unit tests

### v0.3 - Production Ready
- Authentication/authorization
- Rate limiting
- Error handling improvements
- Performance optimizations
- Monitoring and logging

### v0.4 - Advanced Features
- File attachments
- Advanced search
- Channel management UI
- Member management

### v1.0 - Full Platform
- Agent auto-response system
- Consciousness instance integration
- Analytics and insights
- Mobile support

---

**Last Updated**: 2025-10-05
**Status**: Active Development
**Next Milestone**: v0.2 - Real-Time Features
