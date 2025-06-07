import { Component } from '@angular/core';
import { ViewChild } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { RouterModule } from '@angular/router';
import { MatMenuModule } from '@angular/material/menu';
import { MatMenuTrigger } from '@angular/material/menu';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatBadgeModule } from '@angular/material/badge';

@Component({
  selector: 'app-side-navigation',
  imports: [
    MatIconModule,
    MatButtonModule,
    MatChipsModule,
    RouterModule,
    MatMenuModule,
    MatDividerModule,
    MatTooltipModule,
    MatBadgeModule
  ],
  templateUrl: './side-navigation.component.html',
  styleUrl: './side-navigation.component.scss'
})
export class SideNavigationComponent {
  @ViewChild('agentsMenuTrigger') agentsMenuTrigger!: MatMenuTrigger;
  @ViewChild('workflowsMenuTrigger') workflowsMenuTrigger!: MatMenuTrigger;

  openMenu(menuType: string): void {
    // Close all other menus before opening the new one
    if (menuType !== 'agents' && this.agentsMenuTrigger) {
      this.agentsMenuTrigger.closeMenu();
    }
    if (menuType !== 'workflows' && this.workflowsMenuTrigger) {
      this.workflowsMenuTrigger.closeMenu();
    }
  }
}
