-- Initial schema migration: documents the existing tables
-- These tables are created by ensure_*_tables() functions
-- This migration exists to establish migration tracking baseline

-- Note: We don't CREATE tables here because they already exist
-- from the ensure_*_tables() startup functions.
-- Future migrations will use ALTER TABLE, CREATE INDEX, etc.

SELECT 1; -- no-op baseline migration
