import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatMenuModule } from '@angular/material/menu';
import { MatDividerModule } from '@angular/material/divider';
import { LibraryAgent } from '../../../models/agent.models';

@Component({
  selector: 'app-agent-card',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatButtonModule, MatChipsModule, MatTooltipModule, MatSlideToggleModule, MatMenuModule, MatDividerModule],
  templateUrl: './agent-card.component.html',
  styleUrls: ['./agent-card.component.scss']
})
export class AgentCardComponent {
  @Input() public agent!: LibraryAgent;

  @Output() public select = new EventEmitter<LibraryAgent>();
  @Output() public favoriteToggled = new EventEmitter<LibraryAgent>();
  @Output() public enabledToggled = new EventEmitter<LibraryAgent>();
  @Output() public duplicateClicked = new EventEmitter<LibraryAgent>();
  @Output() public deleteClicked = new EventEmitter<LibraryAgent>();
  @Output() public exportClicked = new EventEmitter<LibraryAgent>();
  @Output() public openClicked = new EventEmitter<LibraryAgent>();
  @Output() public tagClicked = new EventEmitter<string>();

  public onSelect(): void { this.select.emit(this.agent); }
  public onToggleFavorite(event: MouseEvent): void { event.stopPropagation(); this.favoriteToggled.emit(this.agent); }
  public onToggleEnabled(event: MouseEvent): void { event.stopPropagation(); this.enabledToggled.emit(this.agent); }
  public onDuplicate(event: MouseEvent): void { event.stopPropagation(); this.duplicateClicked.emit(this.agent); }
  public onDelete(event: MouseEvent): void { event.stopPropagation(); this.deleteClicked.emit(this.agent); }
  public onExport(event: MouseEvent): void { event.stopPropagation(); this.exportClicked.emit(this.agent); }
  public onOpen(event: MouseEvent): void { event.stopPropagation(); this.openClicked.emit(this.agent); }
  public onTag(event: MouseEvent, tag: string): void { event.stopPropagation(); this.tagClicked.emit(tag); }

  /**
   * Determine agent status based on multiple factors
   */
  public getAgentStatus(): 'running' | 'idle' | 'disabled' | 'error' | 'ready' {
    if (!this.agent.enabled) return 'disabled';
    if (this.agent.instances > 0) return 'running';
    if (!this.agent.envReady) return 'error';
    if (this.agent.lastUsed) return 'ready';
    return 'idle';
  }

  /**
   * Get human-readable status text
   */
  public getStatusText(): string {
    const status = this.getAgentStatus();
    switch (status) {
      case 'running': return 'Active';
      case 'ready': return 'Ready';
      case 'idle': return 'Idle';
      case 'error': return 'Needs setup';
      case 'disabled': return 'Disabled';
      default: return 'Unknown';
    }
  }

  /**
   * Get category-specific icon
   */
  public getCategoryIcon(): string {
    const icons: Record<string, string> = {
      cognitive: 'psychology',
      automation: 'settings_suggest',
      integration: 'hub',
      analytics: 'analytics',
      security: 'shield',
      experimental: 'science'
    };
    return icons[this.agent.category] || 'smart_toy';
  }

  /**
   * Get top 3 capabilities for display
   */
  public getTopCapabilities() {
    return (this.agent.capabilities || []).slice(0, 3);
  }

  /**
   * Calculate agent health percentage based on metrics
   */
  public getHealthPercentage(): number {
    if (!this.agent.performanceMetrics) return 75; // Default
    const { reliability, responsiveness, memoryUsage, cpuUsage } = this.agent.performanceMetrics;

    // Weighted health calculation
    const reliabilityScore = reliability || 85;
    const responsivenessScore = Math.max(0, 100 - (responsiveness / 50)); // Lower is better
    const memoryScore = Math.max(0, 100 - memoryUsage);
    const cpuScore = Math.max(0, 100 - cpuUsage);

    return Math.round(
      (reliabilityScore * 0.4) +
      (responsivenessScore * 0.2) +
      (memoryScore * 0.2) +
      (cpuScore * 0.2)
    );
  }

  /**
   * Get primary action based on agent state
   */
  public getPrimaryActionText(): string {
    const status = this.getAgentStatus();
    switch (status) {
      case 'running': return 'View Logs';
      case 'ready': return 'Run';
      case 'idle': return 'Activate';
      case 'error': return 'Configure';
      case 'disabled': return 'Enable';
      default: return 'Open';
    }
  }

  public getPrimaryActionIcon(): string {
    const status = this.getAgentStatus();
    switch (status) {
      case 'running': return 'article';
      case 'ready': return 'play_arrow';
      case 'idle': return 'power_settings_new';
      case 'error': return 'settings';
      case 'disabled': return 'toggle_off';
      default: return 'open_in_new';
    }
  }

  public getPrimaryActionColor(): string {
    const status = this.getAgentStatus();
    switch (status) {
      case 'running': return 'accent';
      case 'ready': return 'primary';
      case 'error': return 'warn';
      default: return '';
    }
  }

  /**
   * Handle contextual primary action
   */
  public onPrimaryAction(event: MouseEvent): void {
    event.stopPropagation();
    const status = this.getAgentStatus();

    switch (status) {
      case 'disabled':
        this.onToggleEnabled(event);
        break;
      case 'error':
        this.onOpen(event); // Open configuration
        break;
      default:
        this.onOpen(event); // Generic open action
        break;
    }
  }
}


