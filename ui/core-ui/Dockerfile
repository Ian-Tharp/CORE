FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 4200

# Install Angular CLI globally
RUN npm install -g @angular/cli

# Development command (can be overridden)
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200"]