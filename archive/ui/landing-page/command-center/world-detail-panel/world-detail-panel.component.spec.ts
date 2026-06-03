import { of } from 'rxjs';

import { WorldDetailPanelComponent } from './world-detail-panel.component';

describe('WorldDetailPanelComponent', () => {
  it('should call modular world-agent lore workflow and link the generated wiki page', () => {
    // Arrange
    const metadataService = {
      linkWikiPage: jest.fn(),
      onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
      getConnectionsForTile: jest.fn().mockReturnValue([]),
      addConnection: jest.fn()
    };
    const worldsService = {
      generateAgentLore: jest.fn().mockReturnValue(of({
        title: 'Verdant Gate',
        content: '# Verdant Gate',
        generated_by: 'world_lore_architect',
        audit: {
          approved: true,
          confidence: 0.91,
          contradictions: [],
          missing_details: [],
          suggestions: []
        }
      })),
      auditWorldAgent: jest.fn().mockReturnValue(of({
        generated_by: 'canon_continuity_auditor',
        audit: {
          approved: true,
          confidence: 0.88,
          contradictions: [],
          missing_details: [],
          suggestions: []
        }
      })),
      saveAgentLore: jest.fn().mockReturnValue(of({ id: 'wiki-1', title: 'Verdant Gate' })),
      listKnowledge: jest.fn().mockReturnValue(of([])),
      listAssets: jest.fn().mockReturnValue(of([]))
    };
    const spawnTemplates = {
      listTemplates: jest.fn().mockReturnValue(of({
        templates: [{
          id: 'tmpl-world-lore-architect',
          name: 'World Lore Architect',
          role: 'world_lore_architect'
        }],
        count: 1
      }))
    };
    const component = new WorldDetailPanelComponent(
      metadataService as any,
      { listBoards: jest.fn().mockReturnValue([]) } as any,
      { listWiki: jest.fn().mockReturnValue(of([])) } as any,
      worldsService as any,
      spawnTemplates as any,
      {} as any,
      { navigate: jest.fn() } as any
    );
    component.ngOnInit();
    component.worldId = 'world-1';
    component.selectedTile = {
      index: 7,
      x: 0,
      y: 0,
      worldX: 0,
      worldY: 0,
      worldZ: 0,
      terrain: 'plain',
      biome: 'forest',
      resource: 'none'
    };
    component.metadata = { tileIndex: 7, name: 'Verdant Gate', createdAt: '', updatedAt: '', version: 1 };

    // Act
    component.generateLore('lore');

    // Assert
    expect(worldsService.generateAgentLore).toHaveBeenCalledWith('world-1', expect.objectContaining({
      tile_index: 7,
      kind: 'Overview',
      agent_id: 'world_lore_architect'
    }));
    expect(component.loreDraft?.title).toBe('Verdant Gate');
    expect(metadataService.linkWikiPage).not.toHaveBeenCalled();

    component.approveLoreDraft();

    expect(worldsService.saveAgentLore).toHaveBeenCalledWith('world-1', expect.objectContaining({
      tile_index: 7,
      title: 'Verdant Gate'
    }));
    expect(metadataService.linkWikiPage).toHaveBeenCalledWith(7, 'wiki-1');
    expect(component.loreAuditSummary).toContain('91% confidence');
  });

  it('should run canon audit for the selected world', () => {
    // Arrange
    const worldsService = {
      auditWorldAgent: jest.fn().mockReturnValue(of({
        generated_by: 'canon_continuity_auditor',
        audit: {
          approved: true,
          confidence: 0.88,
          contradictions: [],
          missing_details: [],
          suggestions: []
        }
      })),
      listKnowledge: jest.fn().mockReturnValue(of([])),
      listAssets: jest.fn().mockReturnValue(of([]))
    };
    const metadataService = {
      onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
      getConnectionsForTile: jest.fn().mockReturnValue([])
    };
    const component = new WorldDetailPanelComponent(
      metadataService as any,
      { listBoards: jest.fn().mockReturnValue([]) } as any,
      { listWiki: jest.fn().mockReturnValue(of([])) } as any,
      worldsService as any,
      { listTemplates: jest.fn().mockReturnValue(of({ templates: [], count: 0 })) } as any,
      {} as any,
      { navigate: jest.fn() } as any
    );
    component.worldId = 'world-1';
    component.selectedTile = {
      index: 4,
      x: 0,
      y: 0,
      worldX: 0,
      worldY: 0,
      worldZ: 0,
      terrain: 'water',
      biome: 'none',
      resource: 'none'
    };

    // Act
    component.auditCurrentLore();

    // Assert
    expect(worldsService.auditWorldAgent).toHaveBeenCalledWith('world-1', expect.objectContaining({ tile_index: 4 }));
    expect(component.loreAuditSummary).toContain('88% confidence');
  });

  it('should suggest and accept a world connection', () => {
    // Arrange
    const metadataService = {
      onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
      getConnectionsForTile: jest.fn().mockReturnValue([]),
      addConnection: jest.fn()
    };
    const worldsService = {
      suggestWorldConnections: jest.fn().mockReturnValue(of({
        generated_by: 'world_connection_cartographer',
        suggestions: [{
          from_tile_index: 4,
          to_tile_index: 8,
          type: 'trade',
          label: 'Trade',
          rationale: 'Resource complementarity.',
          confidence: 0.82
        }]
      })),
      listKnowledge: jest.fn().mockReturnValue(of([])),
      listAssets: jest.fn().mockReturnValue(of([]))
    };
    const component = new WorldDetailPanelComponent(
      metadataService as any,
      { listBoards: jest.fn().mockReturnValue([]) } as any,
      { listWiki: jest.fn().mockReturnValue(of([])) } as any,
      worldsService as any,
      { listTemplates: jest.fn().mockReturnValue(of({ templates: [], count: 0 })) } as any,
      {} as any,
      { navigate: jest.fn() } as any
    );
    component.worldId = 'world-1';
    component.selectedTile = {
      index: 4,
      x: 0,
      y: 0,
      worldX: 0,
      worldY: 0,
      worldZ: 0,
      terrain: 'water',
      biome: 'none',
      resource: 'none'
    };

    // Act
    component.suggestConnections();
    component.acceptConnectionSuggestion(component.connectionSuggestions[0]);

    // Assert
    expect(worldsService.suggestWorldConnections).toHaveBeenCalledWith('world-1', expect.objectContaining({ tile_index: 4 }));
    expect(metadataService.addConnection).toHaveBeenCalledWith(4, 8, 'trade', true, 'Trade');
  });

  it('should draft an image prompt with the visual director', () => {
    // Arrange
    const worldsService = {
      generateImagePrompt: jest.fn().mockReturnValue(of({
        generated_by: 'visual_prompt_director',
        prompt: 'Wide cinematic portrait of Azure Canopy',
        palette: ['cyan energy'],
        constraints: []
      })),
      listKnowledge: jest.fn().mockReturnValue(of([])),
      listAssets: jest.fn().mockReturnValue(of([]))
    };
    const metadataService = {
      onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
      getConnectionsForTile: jest.fn().mockReturnValue([])
    };
    const component = new WorldDetailPanelComponent(
      metadataService as any,
      { listBoards: jest.fn().mockReturnValue([]) } as any,
      { listWiki: jest.fn().mockReturnValue(of([])) } as any,
      worldsService as any,
      { listTemplates: jest.fn().mockReturnValue(of({ templates: [], count: 0 })) } as any,
      {} as any,
      { navigate: jest.fn() } as any
    );
    component.worldId = 'world-1';
    component.selectedTile = {
      index: 2,
      x: 0,
      y: 0,
      worldX: 0,
      worldY: 0,
      worldZ: 0,
      terrain: 'water',
      biome: 'forest',
      resource: 'none'
    };

    // Act
    component.directArtPrompt();

    // Assert
    expect(worldsService.generateImagePrompt).toHaveBeenCalledWith('world-1', expect.objectContaining({ tile_index: 2 }));
    expect(component.artPrompt).toContain('Azure Canopy');
    expect(component.artPromptStatus).toContain('visual_prompt_director');
  });

  it('should auto-create a mood board from world assets and link it to the tile', () => {
    // Arrange
    const metadataService = {
      onSelectedMetadataChanged: jest.fn().mockReturnValue(of(null)),
      getConnectionsForTile: jest.fn().mockReturnValue([]),
      linkBoard: jest.fn(),
      unlinkBoard: jest.fn()
    };
    const createdBoard = {
      id: 'board-1',
      title: 'Verdant Gate - Mood Board',
      worldId: 'world-1',
      cards: [
        { id: 'card-1', title: 'World Overview', notes: 'Terrain: plain', createdAt: '' }
      ],
      createdAt: ''
    };
    const creativeData = {
      listBoards: jest.fn().mockReturnValue([createdBoard]),
      createBoardWithCards: jest.fn().mockReturnValue(createdBoard),
      getBoard: jest.fn().mockReturnValue(createdBoard)
    };
    const router = { navigate: jest.fn() };
    const component = new WorldDetailPanelComponent(
      metadataService as any,
      creativeData as any,
      { listWiki: jest.fn().mockReturnValue(of([])) } as any,
      { listKnowledge: jest.fn().mockReturnValue(of([])), listAssets: jest.fn().mockReturnValue(of([])) } as any,
      { listTemplates: jest.fn().mockReturnValue(of({ templates: [], count: 0 })) } as any,
      {} as any,
      router as any
    );
    component.worldId = 'world-1';
    component.selectedTile = {
      index: 5,
      x: 0,
      y: 0,
      worldX: 0,
      worldY: 0,
      worldZ: 0,
      terrain: 'plain',
      biome: 'forest',
      resource: 'node'
    };
    component.metadata = {
      tileIndex: 5,
      name: 'Verdant Gate',
      description: 'A living threshold world.',
      createdAt: '',
      updatedAt: '',
      version: 1
    };
    component.worldArt = [{
      id: 'asset-1',
      kind: 'art',
      title: 'World portrait',
      image_b64: 'abc',
      created_at: ''
    }];
    component.inhabitants = [{
      id: 'char-1',
      name: 'Kael Dawnseer',
      image_b64: 'def',
      created_at: '',
      updated_at: ''
    }];
    component.linkedWikiPages = [{
      id: 'wiki-1',
      title: 'Verdant Gate Lore',
      content: 'Ancient gardens orbit the gates.',
      created_at: '',
      updated_at: ''
    }];
    component.artPrompt = 'Solarpunk gate world prompt';

    // Act
    component.autoCreateMoodBoard();

    // Assert
    expect(creativeData.createBoardWithCards).toHaveBeenCalledWith(
      { title: 'Verdant Gate - Mood Board', worldId: 'world-1' },
      expect.arrayContaining([
        expect.objectContaining({ title: 'World Overview' }),
        expect.objectContaining({ title: 'Palette' }),
        expect.objectContaining({ title: 'World Art: World portrait' }),
        expect.objectContaining({ title: 'Inhabitant: Kael Dawnseer' }),
        expect.objectContaining({ title: 'Lore: Verdant Gate Lore' }),
        expect.objectContaining({ title: 'Prompt Direction' })
      ])
    );
    expect(metadataService.linkBoard).toHaveBeenCalledWith(5, 'board-1');
    expect(component.selectedBoard?.id).toBe('board-1');

    component.openBoard(createdBoard);

    expect(router.navigate).toHaveBeenCalledWith(['/creative/boards'], expect.objectContaining({
      queryParams: expect.objectContaining({ boardId: 'board-1', worldId: 'world-1' })
    }));
  });
});
