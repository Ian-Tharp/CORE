import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

export interface RemoteWorldCardModel {
  id: string;
  name: string;
  updated_at: string;
  preview?: string | null;
  origin?: 'human' | 'ai';
}

@Component({
  selector: 'app-remote-world-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './world-card-remote.component.html',
  styleUrl: './world-card-remote.component.scss'
})
export class RemoteWorldCardComponent {
  @Input({ required: true }) world!: RemoteWorldCardModel;
  @Input() showActions = false;
  @Output() delete = new EventEmitter<string>();

  constructor(private readonly router: Router) {}

  openInEditor(): void {
    this.router.navigate(['/command-center'], { queryParams: { worldId: this.world.id } });
  }

  onDelete(ev: MouseEvent): void { ev.stopPropagation(); this.delete.emit(this.world.id); }
}


