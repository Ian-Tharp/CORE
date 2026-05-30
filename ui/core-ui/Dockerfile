FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
# --legacy-peer-deps: @angular/material@19.2.17 pins @angular/cdk@19.2.17 exactly
# while the lockfile resolves cdk@19.2.19 (a benign patch-level mismatch the host
# tree also runs with). Strict `npm ci` rejects it without this flag.
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Expose port
EXPOSE 4200

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Development command (can be overridden)
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200"]