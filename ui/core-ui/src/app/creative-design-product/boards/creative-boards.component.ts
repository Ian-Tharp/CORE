import { Component, inject } from '@angular/core';
import { CommonModule, NgFor } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CreativeDataService, Board } from '../services/creative-data.service';

@Component({
  selector: 'app-creative-boards',
  imports: [CommonModule, FormsModule],
  templateUrl: './creative-boards.component.html',
  styleUrl: './creative-boards.component.scss'
})
export class CreativeBoardsComponent {
  private readonly data = inject(CreativeDataService);
  private readonly route = inject(ActivatedRoute);
  worldId?: string | null = null;
  boards: Board[] = [];
  newTitle = '';

  ngOnInit(): void {
    this.worldId = this.route.snapshot.queryParamMap.get('projectId');
    this.refresh();
  }
  refresh(): void { this.boards = this.data.listBoards(this.worldId || undefined); }
  createBoard(): void {
    if (!this.newTitle.trim()) return;
    this.data.createBoard({ title: this.newTitle.trim(), worldId: this.worldId || undefined });
    this.newTitle = ''; this.refresh();
  }
}
