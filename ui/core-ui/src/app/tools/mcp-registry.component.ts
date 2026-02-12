import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

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
  imports: [CommonModule, FormsModule],
  templateUrl: './mcp-registry.component.html',
  styleUrls: ['./mcp-registry.component.scss']
})
export class McpRegistryComponent implements OnInit {
  registry: MCPRegistry | null = null;
  loading = false;
  error: string | null = null;
  searchQuery = '';
  selectedServer: string | null = null;
  selectedTool: MCPTool | null = null;

  constructor() {}

  ngOnInit(): void {
    this.loadRegistry();
  }

  async loadRegistry(): Promise<void> {
    this.loading = true;
    this.error = null;

    try {
      const response = await fetch('/mcp/registry', {
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to load MCP registry: ${response.statusText}`);
      }

      this.registry = await response.json();
    } catch (err) {
      this.error = err instanceof Error ? err.message : 'Failed to load MCP registry';
      console.error('Error loading MCP registry:', err);
    } finally {
      this.loading = false;
    }
  }

  private getAuthToken(): string {
    // In a real implementation, this would get the auth token from a service
    return localStorage.getItem('auth_token') || '';
  }

  get filteredServers(): MCPServer[] {
    if (!this.registry) return [];
    
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

  get allTools(): MCPTool[] {
    if (!this.registry) return [];
    
    const tools: MCPTool[] = [];
    for (const server of this.filteredServers) {
      tools.push(...server.tools);
    }
    
    return tools;
  }

  get serverNames(): string[] {
    if (!this.registry) return [];
    return this.registry.servers.map(server => server.name);
  }

  selectTool(tool: MCPTool): void {
    this.selectedTool = tool;
  }

  clearSelection(): void {
    this.selectedTool = null;
  }

  getServerStatus(server: MCPServer): 'success' | 'warning' | 'error' {
    switch (server.status) {
      case 'running': return 'success';
      case 'configured': return 'warning';
      default: return 'error';
    }
  }

  getServerStatusText(server: MCPServer): string {
    switch (server.status) {
      case 'running': return 'Running';
      case 'configured': return 'Configured';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  }

  async refreshRegistry(): Promise<void> {
    await this.loadRegistry();
  }
}