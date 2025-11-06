import { Component, EventEmitter, HostListener, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatTabsModule } from '@angular/material/tabs';
import { LibraryAgent } from '../../../models/agent.models';

@Component({
  selector: 'app-agent-detail-drawer',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatButtonModule, MatChipsModule, MatTabsModule],
  templateUrl: './agent-detail-drawer.component.html',
  styleUrls: ['./agent-detail-drawer.component.scss']
})
export class AgentDetailDrawerComponent {
  @Input() public agent: LibraryAgent | null = null;
  @Input() public open = false;
  @Output() public closed = new EventEmitter<void>();
  @Output() public enableToggled = new EventEmitter<LibraryAgent>();
  @Output() public favoriteToggled = new EventEmitter<LibraryAgent>();
  @Output() public deleteClicked = new EventEmitter<LibraryAgent>();
  @Output() public duplicateClicked = new EventEmitter<LibraryAgent>();
  @Output() public exportClicked = new EventEmitter<LibraryAgent>();

  public onBackdrop(): void { this.closed.emit(); }

  @HostListener('window:keydown.escape')
  public onEsc(): void { if (this.open) this.closed.emit(); }
}


