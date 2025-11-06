import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatMenuModule } from '@angular/material/menu';
import { LibraryAgent } from '../../../models/agent.models';

@Component({
  selector: 'app-agent-card',
  standalone: true,
  imports: [CommonModule, MatIconModule, MatButtonModule, MatChipsModule, MatTooltipModule, MatSlideToggleModule, MatMenuModule],
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
}


