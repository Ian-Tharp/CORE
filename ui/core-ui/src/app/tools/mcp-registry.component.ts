import { Component, OnInit, inject } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { AppConfigService } from '../services/config/app-config.service';
import { DiscordBridgeService } from '../services/discord-bridge/discord-bridge.service';
import { DiscordBridgeStatusBadgeComponent } from '../shared/discord-bridge-status-badge/discord-bridge-status-badge.component';

interface MCPTool {
  name: string;
  description: string;
  parameters_schema?: any;
  server: string;
}

interface MCPServer {
  name: string;
  type: string;
  command?: string;
  args?: string[];
  env?: { [key: string]: string };
  status: string;
  tools: MCPTool[];
}

interface MCPRegistry {
  servers: MCPServer[];
  total_tools: number;
}

@Component({
  selector: 'app-mcp-registry',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule, AsyncPipe, DiscordBridgeStatusBadgeComponent],
  templateUrl: './mcp-registry.component.html',
  styleUrls: ['./mcp-registry.component.scss']
})
export class McpRegistryComponent implements OnInit {
  private readonly _config = inject(AppConfigService);
  private readonly _discordBridgeService = inject(DiscordBridgeService);
  private readonly _registryUrl = `${this._config.apiBaseUrl}/mcp/registry`;

  public readonly discordBridgeIndicator$ = this._discordBridgeService.indicator$;
  public registry: MCPRegistry | null = null;
  public loading = false;
  public error: string | null = null;
  public searchQuery = '';
  public selectedServer: string | null = null;
  public selectedTool: MCPTool | null = null;
  public lastLoadedAt: Date | null = null;

  constructor() {}

  public ngOnInit(): void {
    this.loadRegistry();
  }

  public async loadRegistry(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      const response = await fetch(this._registryUrl, {
        headers: {
          'X-API-Key': this._config.apiKey
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to load MCP registry: ${response.statusText}`);
      }

      this.registry = await response.json();
      this.lastLoadedAt = new Date();
    } catch (err) {
      this.error = err instanceof Error ? err.message : 'Failed to load MCP registry';
      console.error('Error loading MCP registry:', err);
    } finally {
      this.loading = false;
    }
  }

  public get filteredServers(): MCPServer[] {
    if (!this.registry) {
      return [];
    }
    
    let servers = this.registry.servers;
    
    if (this.searchQuery) {
      const query = this.searchQuery.toLowerCase();
      servers = servers.filter(server => 
        server.name.toLowerCase().includes(query) ||
        server.tools.some(tool => 
          tool.name.toLowerCase().includes(query) || 
          tool.description.toLowerCase().includes(query)
        )
      );
    }
    
    if (this.selectedServer) {
      servers = servers.filter(server => server.name === this.selectedServer);
    }
    
    return servers;
  }

  public get allTools(): MCPTool[] {
    if (!this.registry) {
      return [];
    }
    
    const tools: MCPTool[] = [];
    for (const server of this.filteredServers) {
      tools.push(...this.getVisibleTools(server));
    }
    
    return tools;
  }

  public get serverNames(): string[] {
    if (!this.registry) {
      return [];
    }

    return this.registry.servers.map(server => server.name);
  }

  public get runningServerCount(): number {
    return this.registry?.servers.filter((server) => server.status === 'running').length ?? 0;
  }

  public get configuredServerCount(): number {
    return this.registry?.servers.filter((server) => server.status === 'configured').length ?? 0;
  }

  public get registryTone(): 'ready' | 'pending' | 'attention' {
    if (this.loading) {
      return 'pending';
    }

    if (!this.registry || this.registry.servers.length === 0) {
      return 'attention';
    }

    if (this.runningServerCount > 0) {
      return 'ready';
    }

    return 'pending';
  }

  public get registryStatusLabel(): string {
    switch (this.registryTone) {
      case 'ready':
        return 'Registry ready';
      case 'pending':
        return 'Registry loading';
      default:
        return 'Registry attention';
    }
  }

  public get registryStatusDetail(): string {
    if (this.loading) {
      return 'Syncing MCP server availability from the CORE backend.';
    }

    if (this.error) {
      return 'Registry refresh failed. Retry to restore tool visibility.';
    }

    if (!this.registry || this.registry.servers.length === 0) {
      return 'No MCP servers are configured yet.';
    }

    if (this.runningServerCount === this.registry.servers.length) {
      return 'All configured MCP servers are currently reachable.';
    }

    if (this.runningServerCount > 0) {
      return `${this.runningServerCount} of ${this.registry.servers.length} servers are running.`;
    }

    if (this.configuredServerCount > 0) {
      return `${this.configuredServerCount} server(s) are configured but not running yet.`;
    }

    return 'Registry loaded without any running servers.';
  }

  public get lastLoadedTimeLabel(): string {
    return this.lastLoadedAt
      ? this.lastLoadedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : 'Waiting for first sync';
  }

  public get filteredServerCount(): number {
    return this.filteredServers.length;
  }

  public selectTool(tool: MCPTool): void {
    this.selectedTool = tool;
  }

  public clearSelection(): void {
    this.selectedTool = null;
  }

  public getServerStatus(server: MCPServer): 'success' | 'warning' | 'error' {
    switch (server.status) {
      case 'running': return 'success';
      case 'configured': return 'warning';
      default: return 'error';
    }
  }

  public getServerStatusText(server: MCPServer): string {
    switch (server.status) {
      case 'running': return 'Running';
      case 'configured': return 'Configured';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  }

  public getVisibleTools(server: MCPServer): MCPTool[] {
    if (!this.searchQuery) {
      return server.tools;
    }

    const query = this.searchQuery.toLowerCase();
    return server.tools.filter((tool) =>
      tool.name.toLowerCase().includes(query) ||
      tool.description.toLowerCase().includes(query)
    );
  }

  public trackByServer(_index: number, server: MCPServer): string {
    return server.name;
  }

  public trackByTool(_index: number, tool: MCPTool): string {
    return `${tool.server}-${tool.name}`;
  }

  public async refreshRegistry(): Promise<void> {
    await this.loadRegistry();
  }
}