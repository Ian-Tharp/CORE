import { Component, ViewChild, inject } from '@angular/core';
import { AsyncPipe } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { RouterModule } from '@angular/router';
import { MatMenuModule } from '@angular/material/menu';
import { MatMenuTrigger } from '@angular/material/menu';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatBadgeModule } from '@angular/material/badge';
import {
  DiscordBridgeIndicator,
  DiscordBridgeService,
} from '../../services/discord-bridge/discord-bridge.service';
import { DiscordBridgeStatusBadgeComponent } from '../discord-bridge-status-badge/discord-bridge-status-badge.component';

@Component({
  selector: 'app-side-navigation',
  imports: [
    AsyncPipe,
    MatIconModule,
    MatButtonModule,
    MatChipsModule,
    RouterModule,
    MatMenuModule,
    MatDividerModule,
    MatTooltipModule,
    MatBadgeModule,
    DiscordBridgeStatusBadgeComponent
  ],
  templateUrl: './side-navigation.component.html',
  styleUrl: './side-navigation.component.scss'
})
export class SideNavigationComponent {
  private readonly _discordBridgeService = inject(DiscordBridgeService);

  @ViewChild('agentsMenuTrigger') public agentsMenuTrigger!: MatMenuTrigger;
  @ViewChild('workflowsMenuTrigger') public workflowsMenuTrigger!: MatMenuTrigger;

  public readonly discordBridgeIndicator$ = this._discordBridgeService.indicator$;

  public openMenu(menuType: string): void {
    // Close all other menus before opening the new one
    if (menuType !== 'agents' && this.agentsMenuTrigger) {
      this.agentsMenuTrigger.closeMenu();
    }
    if (menuType !== 'workflows' && this.workflowsMenuTrigger) {
      this.workflowsMenuTrigger.closeMenu();
    }
  }

  public getToolsTooltip(indicator: DiscordBridgeIndicator | null): string {
    if (!indicator) {
      return 'Tools & Integrations';
    }

    return `Tools & Integrations · Discord Gateway ${indicator.label}`;
  }
}
