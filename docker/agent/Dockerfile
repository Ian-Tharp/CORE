# OpenClaw Agent Dockerfile for CORE Orchestration
# Multi-stage build optimized for minimal runtime image
FROM node:22-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    git \
    python3 \
    make \
    g++ \
    curl

# Install Bun for build scripts (required by OpenClaw)
RUN curl -fsSL https://bun.sh/install | ash -s -- -y
ENV PATH="/root/.bun/bin:${PATH}"

# Enable corepack for pnpm
RUN corepack enable

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY ui/package.json ./ui/package.json
COPY patches ./patches
COPY scripts ./scripts

# Install dependencies
RUN pnpm install --frozen-lockfile --production=false

# Copy source and build
COPY . .
RUN OPENCLAW_A2UI_SKIP_MISSING=1 pnpm build

# Build UI with pnpm (more reliable for agent deployments)
ENV OPENCLAW_PREFER_PNPM=1
RUN pnpm ui:build

# Runtime stage - minimal Alpine image
FROM node:22-alpine AS runtime

# Install runtime dependencies only
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    dumb-init \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1000 openclaw \
    && adduser -D -s /bin/sh -u 1000 -G openclaw openclaw

# Create necessary directories
RUN mkdir -p /app /home/openclaw/.openclaw /home/openclaw/workspace \
    && chown -R openclaw:openclaw /app /home/openclaw

WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=openclaw:openclaw /app/dist ./dist
COPY --from=builder --chown=openclaw:openclaw /app/package.json ./
COPY --from=builder --chown=openclaw:openclaw /app/node_modules ./node_modules
COPY --from=builder --chown=openclaw:openclaw /app/skills ./skills
COPY --from=builder --chown=openclaw:openclaw /app/docs ./docs

# Set up environment
ENV NODE_ENV=production
ENV HOME=/home/openclaw
ENV OPENCLAW_WORKSPACE_DIR=/home/openclaw/workspace

# Switch to non-root user
USER openclaw

# Health check for CORE to monitor agent instances
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node dist/index.js health --gateway-url "ws://localhost:18789" || exit 1

# Expose default gateway port
EXPOSE 18789

# Use dumb-init to properly handle signals
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js", "gateway", "--bind", "lan", "--port", "18789"]