-- Webhook registrations and delivery history
-- Persists webhook endpoints and tracks delivery attempts for reliability

-- =============================================================================
-- WEBHOOK REGISTRATIONS: Persistent webhook endpoint configuration
-- =============================================================================
CREATE TABLE IF NOT EXISTS webhook_registrations (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,
    secret VARCHAR(512),
    headers JSONB DEFAULT '{}',
    max_retries INTEGER DEFAULT 3,
    is_active BOOLEAN DEFAULT TRUE,
    delivery_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_delivery_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_webhook_registrations_active
    ON webhook_registrations(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_webhook_registrations_created
    ON webhook_registrations(created_at DESC);

-- =============================================================================
-- WEBHOOK DELIVERIES: Delivery attempt history for debugging and monitoring
-- =============================================================================
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id VARCHAR(36) PRIMARY KEY,
    webhook_id VARCHAR(36) NOT NULL REFERENCES webhook_registrations(id) ON DELETE CASCADE,
    event VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    attempts INTEGER DEFAULT 0,
    status_code INTEGER,
    response_body TEXT,
    error TEXT,
    delivered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_id
    ON webhook_deliveries(webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_created
    ON webhook_deliveries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_event
    ON webhook_deliveries(event);

-- Grant permissions
GRANT ALL PRIVILEGES ON webhook_registrations TO core_user;
GRANT ALL PRIVILEGES ON webhook_deliveries TO core_user;