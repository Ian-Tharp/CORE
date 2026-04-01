import { Component, Input } from '@angular/core';

import { DiscordBridgeIndicator } from '../../services/discord-bridge/discord-bridge.service';

@Component({
  selector: 'app-discord-bridge-status-badge',
  standalone: true,
  templateUrl: './discord-bridge-status-badge.component.html',
  styleUrl: './discord-bridge-status-badge.component.scss'
})
export class DiscordBridgeStatusBadgeComponent {
  @Input() public indicator: DiscordBridgeIndicator | null = null;
  @Input() public variant: 'chip' | 'dot' = 'chip';
  @Input() public labelPrefix = 'Discord Gateway';

  public get ariaLabel(): string {
    return `${this.labelPrefix}: ${this.indicator?.label ?? 'Unknown'}`;
  }

  public get toneClass(): string {
    return this.indicator?.tone ?? 'unknown';
  }
}
