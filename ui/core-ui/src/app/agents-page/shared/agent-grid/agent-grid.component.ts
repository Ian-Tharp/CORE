import { Component, EventEmitter, Input, Output, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LibraryAgent } from '../../../models/agent.models';
import { AgentCardComponent } from '../agent-card/agent-card.component';

@Component({
  selector: 'app-agent-grid',
  standalone: true,
  imports: [CommonModule, AgentCardComponent],
  templateUrl: './agent-grid.component.html',
  styleUrls: ['./agent-grid.component.scss']
})
export class AgentGridComponent {
  @Input() public set agents(value: LibraryAgent[]) { this._agents.set(value ?? []); }
  @Input() public view: 'grid' | 'list' = 'grid';
  @Output() public select = new EventEmitter<LibraryAgent>();
  @Output() public favoriteToggled = new EventEmitter<LibraryAgent>();
  @Output() public enabledToggled = new EventEmitter<LibraryAgent>();
  @Output() public duplicateClicked = new EventEmitter<LibraryAgent>();
  @Output() public deleteClicked = new EventEmitter<LibraryAgent>();
  @Output() public exportClicked = new EventEmitter<LibraryAgent>();
  @Output() public tagClicked = new EventEmitter<string>();

  private _agents = signal<LibraryAgent[]>([]);
  public readonly list = computed(() => this._agents());
}


