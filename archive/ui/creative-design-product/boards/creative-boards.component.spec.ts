import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap } from '@angular/router';
import { of } from 'rxjs';

import { CreativeBoardsComponent } from './creative-boards.component';
import { CreativeDataService } from '../services/creative-data.service';
import { WorldsService } from '../../services/worlds/worlds.service';
import { CreativeService } from '../../services/creative/creative.service';

describe('CreativeBoardsComponent', () => {
  let component: CreativeBoardsComponent;
  let fixture: ComponentFixture<CreativeBoardsComponent>;
  const board = {
    id: 'board-1',
    title: 'Verdant Gate - Mood Board',
    worldId: 'world-1',
    cards: [
      { id: 'card-1', title: 'World Overview', notes: 'Terrain: forest', createdAt: '' },
      {
        id: 'card-2',
        title: 'World Art: Verdant Gate',
        imageSource: 'world_asset' as const,
        sourceId: 'asset-1',
        createdAt: ''
      },
      {
        id: 'card-3',
        title: 'Inhabitant: Kael',
        imageSource: 'character' as const,
        sourceId: 'char-1',
        createdAt: ''
      }
    ],
    createdAt: ''
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreativeBoardsComponent],
      providers: [
        {
          provide: CreativeDataService,
          useValue: {
            listBoards: jest.fn().mockReturnValue([board]),
            deleteBoard: jest.fn().mockReturnValue(true),
            createBoard: jest.fn().mockReturnValue({
              id: 'b1',
              title: 'Board',
              cards: [],
              createdAt: new Date().toISOString()
            })
          }
        },
        {
          provide: ActivatedRoute,
          useValue: {
            snapshot: { queryParamMap: convertToParamMap({ worldId: 'world-1', boardId: 'board-1' }) },
            params: of({}),
            queryParamMap: of(convertToParamMap({}))
          }
        },
        {
          provide: WorldsService,
          useValue: {
            listAssets: jest.fn().mockReturnValue(of([{
              id: 'asset-1',
              kind: 'art',
              image_b64: 'asset-b64',
              created_at: ''
            }]))
          }
        },
        {
          provide: CreativeService,
          useValue: {
            listCharacters: jest.fn().mockReturnValue(of([{
              id: 'char-1',
              name: 'Kael',
              image_b64: 'char-b64',
              created_at: '',
              updated_at: ''
            }]))
          }
        }
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(CreativeBoardsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should select board from route query params', () => {
    expect(component.worldId).toBe('world-1');
    expect(component.selectedBoard?.id).toBe('board-1');
  });

  it('should resolve source-based board card images from loaded world assets and characters', () => {
    const artCard = board.cards[1];
    const characterCard = board.cards[2];

    expect(component.cardImage(artCard)).toBe('data:image/png;base64,asset-b64');
    expect(component.cardImage(characterCard)).toBe('data:image/png;base64,char-b64');
  });

  it('should delete the selected board and clear selection', () => {
    const data = TestBed.inject(CreativeDataService) as unknown as { deleteBoard: jest.Mock };
    jest.spyOn(window, 'confirm').mockReturnValue(true);

    component.deleteBoard(board);

    expect(data.deleteBoard).toHaveBeenCalledWith('board-1');
    expect(component.selectedBoard).toBeNull();
  });
});
