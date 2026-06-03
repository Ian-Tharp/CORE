import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { Board, BoardCard, CreativeDataService } from '../services/creative-data.service';
import { WorldsService, WorldAsset } from '../../services/worlds/worlds.service';
import { CharacterDto, CreativeService } from '../../services/creative/creative.service';

@Component({
  selector: 'app-creative-boards',
  imports: [CommonModule, FormsModule],
  templateUrl: './creative-boards.component.html',
  styleUrl: './creative-boards.component.scss'
})
export class CreativeBoardsComponent {
  private readonly data = inject(CreativeDataService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly worlds = inject(WorldsService);
  private readonly creative = inject(CreativeService);
  worldId?: string | null = null;
  selectedBoardId?: string | null = null;
  selectedBoard: Board | null = null;
  boards: Board[] = [];
  worldAssets: WorldAsset[] = [];
  characters: CharacterDto[] = [];
  isLoadingSources = false;
  newTitle = '';

  ngOnInit(): void {
    this.worldId = this.route.snapshot.queryParamMap.get('worldId') ?? this.route.snapshot.queryParamMap.get('projectId');
    this.selectedBoardId = this.route.snapshot.queryParamMap.get('boardId');
    this.refresh();
    this.loadSourceImages();
  }
  refresh(): void {
    this.boards = this.data.listBoards(this.worldId || undefined);
    this.selectedBoard = this.selectedBoardId
      ? this.boards.find(board => board.id === this.selectedBoardId) ?? null
      : null;
  }
  selectBoard(board: Board): void {
    this.selectedBoardId = board.id;
    this.selectedBoard = board;
    this.router.navigate(['/creative/boards'], {
      queryParams: {
        worldId: this.worldId || undefined,
        projectId: this.worldId || undefined,
        boardId: board.id
      },
      replaceUrl: true
    });
  }
  createBoard(): void {
    if (!this.newTitle.trim()) {return;}
    const board = this.data.createBoard({ title: this.newTitle.trim(), worldId: this.worldId || undefined });
    this.selectedBoardId = board.id;
    this.newTitle = ''; this.refresh();
  }

  deleteBoard(board: Board, event?: Event): void {
    event?.stopPropagation();
    if (!confirm(`Delete mood board "${board.title}"?`)) {return;}
    const wasSelected = this.selectedBoardId === board.id;
    this.data.deleteBoard(board.id);
    if (wasSelected) {
      this.selectedBoardId = null;
      this.selectedBoard = null;
      this.router.navigate(['/creative/boards'], {
        queryParams: {
          worldId: this.worldId || undefined,
          projectId: this.worldId || undefined
        },
        replaceUrl: true
      });
    }
    this.refresh();
  }

  get boardCountLabel(): string {
    return `${this.boards.length} board${this.boards.length === 1 ? '' : 's'}`;
  }

  get selectedCardCountLabel(): string {
    const count = this.selectedBoard?.cards.length ?? 0;
    return `${count} card${count === 1 ? '' : 's'}`;
  }

  get heroCard(): BoardCard | null {
    return this.selectedBoard?.cards.find(card => !!card.imageUrl) ?? this.selectedBoard?.cards[0] ?? null;
  }

  cardKind(card: BoardCard): string {
    const title = card.title.toLowerCase();
    if (title.includes('palette')) {return 'Palette';}
    if (title.includes('world art')) {return 'World Art';}
    if (title.includes('inhabitant')) {return 'Inhabitant';}
    if (title.includes('lore')) {return 'Lore';}
    if (title.includes('prompt')) {return 'Prompt';}
    return 'Overview';
  }

  cardSourceLabel(card: BoardCard): string {
    if (card.imageSource === 'world_asset') {return 'World Asset';}
    if (card.imageSource === 'character') {return 'Character';}
    return this.cardKind(card);
  }

  paletteItems(card: BoardCard): string[] {
    if (this.cardKind(card) !== 'Palette') {return [];}
    return (card.notes ?? '')
      .split('\n')
      .map(line => line.replace(/^-\s*/, '').trim())
      .filter(Boolean);
  }

  cardImage(card: BoardCard): string | null {
    if (card.imageUrl) {return card.imageUrl;}
    if (card.imageSource === 'world_asset' && card.sourceId) {
      const asset = this.worldAssets.find(item => item.id === card.sourceId);
      return asset ? `data:image/png;base64,${asset.image_b64}` : null;
    }
    if (card.imageSource === 'character' && card.sourceId) {
      const character = this.characters.find(item => item.id === card.sourceId);
      return character?.image_b64 ? `data:image/png;base64,${character.image_b64}` : null;
    }
    return null;
  }

  hasImageReference(card: BoardCard): boolean {
    return !!card.imageSource && !!card.sourceId && !card.imageUrl;
  }

  isCardImageLoading(card: BoardCard): boolean {
    return this.isLoadingSources && this.hasImageReference(card) && !this.cardImage(card);
  }

  backToCommandCenter(): void {
    this.router.navigate(['/command-center/edit'], {
      queryParams: { worldId: this.worldId || undefined }
    });
  }

  private loadSourceImages(): void {
    if (!this.worldId) {return;}
    this.isLoadingSources = true;
    let pending = 2;
    const complete = () => {
      pending -= 1;
      if (pending <= 0) {this.isLoadingSources = false;}
    };
    this.worlds.listAssets(this.worldId).subscribe({
      next: (assets) => { this.worldAssets = assets ?? []; complete(); },
      error: () => { this.worldAssets = []; complete(); }
    });
    this.creative.listCharacters(this.worldId).subscribe({
      next: (characters) => { this.characters = characters ?? []; complete(); },
      error: () => { this.characters = []; complete(); }
    });
  }
}
